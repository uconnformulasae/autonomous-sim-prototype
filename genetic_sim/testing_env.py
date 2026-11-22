"""
@file testing_env.py
@brief Race a population of randomly initialised neural nets on one track.

Creates n NNControllers (each with its own random weights), saves every net's
weights and biases, then races all n cars at once on the same track from the
same start pose and records how each did.

The cars are "ghosts" to each other (see make_ghost_env): car-to-car collisions
are off and no car sees the others in its LiDAR, so every car drives exactly as
it would alone. A crashed car is stopped and left where it is.

Outputs, in runs/<track>_<timestamp>/:
    weights.npz  params      (n, P) flat weights+biases per car, as MLP.get_params()
                 layer_sizes layer sizes of the MLP, to rebuild it
                 seeds       weight-init seed per car
    results.csv  car, seed, score, crashed, death_time_s, sim_time_s, distance_m,
                 median_speed_mps, laps

    death_time_s is the sim time of the crash, empty if the car never crashed
    (it finished 2 laps or hit --max-time).

Rendering is on by default: the camera follows the leading car (orange),
running cars are purple and crashed cars grey. When the race ends the window
stays open until you close it. Closing it mid-race stops the race early and
still saves the results.

    python testing_env.py -n 20
    python testing_env.py -n 50 --map Monza --max-time 30
    python testing_env.py -n 200 --headless                  # no window, much faster
    python testing_env.py --load runs/<run>/weights.npz      # re-race saved nets
"""

import argparse
import csv
import os
import time
import warnings
from multiprocessing import Pool

import gym
import numpy as np

from f110_gym.envs.base_classes import Integrator, RaceCar
from nn_controller import HERE, NNController, load_tracks
from scoring import ScoreKeeper


MAP = "Budapest"
NUM_AGENTS = 10
TIMESTEP = 0.01  # s, env physics step

# Car colours in the render window (RGB).
LEADER_COLOUR = (255, 140, 0)
ALIVE_COLOUR = (172, 97, 185)
DEAD_COLOUR = (110, 110, 110)


def is_window_closed(ex):
    """@brief Whether an exception means the render window was closed.

    @param ex The exception caught from env.render().
    @return bool True if the user closed the window.
    """
    # EnvRenderer.on_close raises a plain Exception with this message.
    return 'window was closed' in str(ex)


def add_race_view(env):
    """@brief Register a render callback: follow the leader, colour cars by state, show status.

    @param env The gym env; race() must set env.unwrapped.view (alive, distance, time, score).
    """
    def render_callback(e):
        v = env.unwrapped.view
        alive, distance = v['alive'], v['distance']
        leader = int(np.argmax(np.where(alive, distance, -1))) if alive.any() else int(np.argmax(distance))

        for i, car in enumerate(e.cars):
            colour = LEADER_COLOUR if i == leader else ALIVE_COLOUR if alive[i] else DEAD_COLOUR
            car.colors = list(colour) * 4

        x = e.cars[leader].vertices[::2]
        y = e.cars[leader].vertices[1::2]
        top, bottom, left, right = max(y), min(y), min(x), max(x)
        e.score_label.text = (f'Time: {v["time"]:.2f}s  Running: {alive.sum()}/{len(alive)}  '
                              f'Leader: car {leader} ({distance[leader]:.1f} m, score {v["score"][leader]:.0f})')
        e.score_label.x = left
        e.score_label.y = top - 700
        e.left, e.right = left - 800, right + 800
        e.top, e.bottom = top + 800, bottom - 800
    env.add_render_callback(render_callback)


def make_population(n, base_seed):
    """@brief Make n controllers, each seeded base_seed + i so a run can be reproduced.

    @param n         Population size.
    @param base_seed Seed of car 0.
    @return tuple (controllers, seeds).
    """
    seeds = [base_seed + i for i in range(n)]
    return [NNController(seed=s) for s in seeds], seeds


def load_population(path):
    """@brief Load a population from a weights.npz file.

    @param path Path to the .npz written by save_weights().
    @return tuple (controllers, seeds).
    """
    data = np.load(path)
    controllers = []
    for params in data['params']:
        c = NNController(hidden=tuple(data['layer_sizes'][1:-1]))
        c.net.set_params(params)
        controllers.append(c)
    return controllers, list(data['seeds'])


def save_weights(path, controllers, seeds):
    """@brief Save a population as weights.npz (params, layer_sizes, seeds).

    @param path        Output .npz path.
    @param controllers The population.
    @param seeds       Weight-init seed of each car.
    """
    first = controllers[0].net
    layer_sizes = [first.weights[0].shape[0]] + [W.shape[1] for W in first.weights]
    np.savez(path,
             params=np.stack([c.net.get_params() for c in controllers]),
             layer_sizes=np.array(layer_sizes),
             seeds=np.array(seeds))


def make_ghost_env(map_path, n):
    """@brief Make an n-car env where the cars cannot collide with or see each other.

    @param map_path Map path without extension.
    @param n        Number of cars.
    @return The gym env.
    """
    # RaceCar keeps one LiDAR simulator (and so one map) for the whole process,
    # and every new env loads its map into it. Clearing it first gives this env
    # a simulator of its own, which race() swaps back in, so envs for different
    # tracks can live in one process without racing on each other's maps.
    RaceCar.scan_simulator = None
    env = gym.make('f110_gym:f110-v0', map=map_path, map_ext='.png',
                   num_agents=n, timestep=TIMESTEP, integrator=Integrator.RK4)
    env.unwrapped.scan_simulator = RaceCar.scan_simulator
    sim = env.unwrapped.sim

    # Car-to-car collisions: replace the GJK check with "nobody touching".
    # Wall collisions are checked separately per car and still apply.
    def no_car_collisions():
        sim.collisions = np.zeros(sim.num_agents)
        sim.collision_idx = -1 * np.ones(sim.num_agents)
    sim.check_collision = no_car_collisions

    # LiDAR: skip ray-casting the other cars into each car's scan.
    for agent in sim.agents:
        agent.ray_cast_agents = lambda scan: scan
        _freeze_on_crash(agent)
    return env


def _freeze_on_crash(agent):
    """@brief Stop simulating a car once it crashes.

    A crashed car sits still and nothing reads its scan (the scorer only looks
    at living cars), so re-running its physics and 1080-beam LiDAR every step
    is wasted work. After the first crash its last scan is replayed instead.

    @param agent The RaceCar to patch.
    """
    update_pose, update_scan, reset = agent.update_pose, agent.update_scan, agent.reset
    frozen = {'scan': None}

    def patched_update_pose(steer, vel):
        if frozen['scan'] is not None:
            return frozen['scan']
        return update_pose(steer, vel)

    def patched_update_scan(agent_scans, idx):
        if frozen['scan'] is not None:
            agent_scans[idx] = frozen['scan']
            return
        update_scan(agent_scans, idx)
        if agent.in_collision:
            frozen['scan'] = agent_scans[idx]

    def patched_reset(pose):
        frozen['scan'] = None
        reset(pose)

    agent.update_pose = patched_update_pose
    agent.update_scan = patched_update_scan
    agent.reset = patched_reset


def race(env, controllers, start_pose, max_steps, render):
    """@brief Drive all cars at once until every car has crashed or time runs out.

    @param env         Env from make_ghost_env().
    @param controllers One NNController per car.
    @param start_pose  (x, y, theta) shared by all cars.
    @param max_steps   Step limit.
    @param render      Whether to render.
    @return tuple (results, window_open). window_open is False if the render
            window was closed, which also ends the race early.
    """
    n = len(controllers)
    RaceCar.scan_simulator = env.unwrapped.scan_simulator  # this env's map; see make_ghost_env
    obs, _, _, _ = env.reset(np.array([start_pose] * n))
    for c in controllers:
        c.reset()

    scorer = ScoreKeeper(n, obs, TIMESTEP)
    alive = scorer.alive
    # Shared with the render callback; the arrays are updated in place.
    view = env.unwrapped.view = {'alive': alive, 'distance': scorer.distance, 'time': 0.0,
                                 'score': scorer.score}

    sim_time = 0.0
    steps = 0
    # The env's own done flag only tracks car 0, so it is ignored here.
    while alive.any() and steps < max_steps:
        actions = np.zeros((n, 2))  # crashed cars get [0, 0] and stay put
        for i in np.flatnonzero(alive):
            actions[i] = controllers[i].plan(obs, agent=i)
        obs, step_reward, _, _ = env.step(actions)
        sim_time += step_reward
        steps += 1

        scorer.step(obs, sim_time)
        if render:
            view['time'] = sim_time
            try:
                env.render(mode='human')
            except Exception as ex:
                if not is_window_closed(ex):
                    raise
                print(f'Window closed at {sim_time:.2f}s; stopping the race early.')
                render = False
                break

    return scorer.results(obs, sim_time), render


def show_best(params, hidden, track, all_tracks, max_steps):
    """@brief Render one car on one track; the window stays up until closed.

    @param params     Flat weight vector of the car.
    @param hidden     Hidden layer sizes.
    @param track      Track name.
    @param all_tracks Result of load_tracks().
    @param max_steps  Step limit.
    """
    print(f'\nShowing the final best car on {track}. Close the window to exit.')
    c = NNController(hidden=hidden)
    c.net.set_params(params)
    map_path, start_pose = all_tracks[track]
    env = make_ghost_env(map_path, 1)
    add_race_view(env)
    (r,), window_open = race(env, [c], start_pose, max_steps, render=True)
    print(f'{track}: score {r["score"]:.1f}  {"crashed" if r["crashed"] else "survived"}  '
          f'dist={r["distance_m"]:.1f}m  laps={r["laps"]}')
    if window_open:
        try:
            while True:
                env.render(mode='human')
        except Exception as ex:
            if not is_window_closed(ex):
                raise


def write_population(path, params, hidden):
    """@brief Save flat parameter vectors (one row per car) in the weights.npz format.

    @param path   Output .npz path.
    @param params Array (cars, P) of flat weight vectors.
    @param hidden Hidden layer sizes.
    """
    controllers = []
    for p in params:
        c = NNController(hidden=hidden)
        c.net.set_params(p)
        controllers.append(c)
    save_weights(path, controllers, list(range(len(controllers))))


def best_from_saved(path):
    """@brief Find the best car in a saved weights.npz (by results.csv next to it, if any).

    @param path Path to weights.npz or best.npz.
    @return tuple (flat params of the best car, hidden layer sizes).
    """
    controllers, _ = load_population(path)
    hidden = tuple(np.load(path)['layer_sizes'][1:-1])
    best = 0
    results_csv = os.path.join(os.path.dirname(path), 'results.csv')
    if len(controllers) > 1 and os.path.exists(results_csv):
        with open(results_csv, newline='') as f:
            scores = [float(r['score']) for r in csv.DictReader(f)]
        best = int(np.argmax(scores))
    return controllers[best].net.get_params(), hidden


# Each worker process keeps its envs between calls so a track's map is only
# loaded once per group. The cache is dropped when the group changes.
_envs = {}
_envs_group = None


def _quiet_worker():
    """@brief Pool initializer: silence the repeated RK4 integrator warning."""
    # Every env build warns that RK4 is the integrator; once is plenty.
    warnings.filterwarnings('ignore', message='Chosen integrator is RK4')


def _race_track(job):
    """@brief Race a chunk of the population on one track. Runs in a worker process.

    @param job Tuple (group, track, lo, map_path, start_pose, params, hidden, max_steps).
    @return tuple (track, lo, results).
    """
    global _envs_group
    group, track, lo, map_path, start_pose, params, hidden, max_steps = job
    if group != _envs_group:
        _envs.clear()
        _envs_group = group
    n = len(params)
    if (map_path, n) not in _envs:
        _envs[(map_path, n)] = make_ghost_env(map_path, n)
    env = _envs[(map_path, n)]

    controllers = []
    for p in params:
        c = NNController(hidden=hidden)
        c.net.set_params(p)
        controllers.append(c)
    results, _ = race(env, controllers, start_pose, max_steps, render=False)
    return track, lo, results


def choose_workers(requested, n_races):
    """@brief Pick the worker count: `requested`, or 75% of the CPU threads when 0 (at least 1).

    More workers than car-races would only sit idle.

    @param requested Requested workers (0 for the default).
    @param n_races   Number of car-races available to share.
    @return int Worker count.
    """
    default = max(1, (os.cpu_count() or 1) * 3 // 4)
    return min(requested or default, n_races)


def make_pool(workers):
    """@brief Create the worker process pool.

    @param workers Number of processes.
    @return multiprocessing.Pool
    """
    return Pool(workers, initializer=_quiet_worker)


def split_jobs(n_cars, n_tracks, workers):
    """@brief Slices (lo, hi) of the population that each track is split into.

    The work is n_cars * n_tracks car-races, shared as evenly as possible over
    the workers: each job races cars_per_job cars on one track (a job cannot
    span tracks, since it runs in one env).

    @param n_cars   Population size.
    @param n_tracks Tracks raced at once.
    @param workers  Worker processes.
    @return list of (lo, hi) pairs.
    """
    cars_per_job = -(-n_cars * n_tracks // workers)  # ceil
    bounds = np.linspace(0, n_cars, -(-n_cars // cars_per_job) + 1).astype(int)
    return list(zip(bounds[:-1], bounds[1:]))


def race_tracks(pool, group, tracks, all_tracks, params, hidden, max_steps, workers):
    """@brief Race every car on every track.

    The cars are ghosts to each other, so splitting them across processes does
    not change any car's result, only how many cores share the work.

    @param pool       Worker pool from make_pool().
    @param group      Cache key; worker env caches reset when it changes.
    @param tracks     Track names to race.
    @param all_tracks Result of load_tracks().
    @param params     Array (cars, P) of flat weight vectors.
    @param hidden     Hidden layer sizes.
    @param max_steps  Step limit per race.
    @param workers    Worker processes.
    @return dict {track: [result dict per car]}.
    """
    jobs = [(group, t, lo, *all_tracks[t], params[lo:hi], hidden, max_steps)
            for t in tracks for lo, hi in split_jobs(len(params), len(tracks), workers)]
    per_track = {t: [None] * len(params) for t in tracks}
    for t, lo, results in pool.imap_unordered(_race_track, jobs):
        per_track[t][lo:lo + len(results)] = results
    return per_track


def main():
    """@brief Command-line entry point: race n nets on one track and save the results."""
    ###################
    # CONFIGURE
    map = MAP
    n = NUM_AGENTS
    render = True
    ###################
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, default=n, help='number of neural nets')
    parser.add_argument('--map', default=map, help='track name from start_poses.csv')
    parser.add_argument('--seed', type=int, default=0, help='base seed for weight init')
    parser.add_argument('--max-time', type=float, default=60.0, help='sim seconds per car before stopping it')
    parser.add_argument('--load', help='race nets from a saved weights.npz instead of new random ones')
    parser.add_argument('--headless', action='store_true', help='no render window (faster)')
    args = parser.parse_args()
    render = render and not args.headless

    tracks = load_tracks()
    if args.map not in tracks:
        parser.error(f'unknown track {args.map!r}; choose from: {", ".join(tracks)}')
    map_path, start_pose = tracks[args.map]

    if args.load:
        controllers, seeds = load_population(args.load)
    else:
        controllers, seeds = make_population(args.n, args.seed)

    run_dir = os.path.join(HERE, 'runs', f'{args.map.replace(" ", "")}_{time.strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(run_dir)
    save_weights(os.path.join(run_dir, 'weights.npz'), controllers, seeds)

    env = make_ghost_env(map_path, len(controllers))
    if render:
        add_race_view(env)

    max_steps = int(args.max_time / TIMESTEP)
    raced, window_open = race(env, controllers, start_pose, max_steps, render)
    results = [{'car': i, 'seed': seed, **r} for i, (seed, r) in enumerate(zip(seeds, raced))]
    for r in results:
        print(f"car {r['car']:3d}  score={r['score']:8.1f}  {'crashed' if r['crashed'] else 'survived':8s}  "
              f"t={r['sim_time_s']:7.2f}s  dist={r['distance_m']:7.2f}m  "
              f"median v={r['median_speed_mps']:5.2f}m/s  laps={r['laps']}")

    with open(os.path.join(run_dir, 'results.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    best = max(results, key=lambda r: r['score'])
    print(f"\nBest: car {best['car']} scored {best['score']} ({best['distance_m']} m). Saved to {run_dir}")

    # Keep the final frame up until the window is closed.
    if window_open:
        print('Race over. Close the render window to exit.')
        try:
            while True:
                env.render(mode='human')
        except Exception as ex:
            if not is_window_closed(ex):
                raise

if __name__ == '__main__':
    main()
