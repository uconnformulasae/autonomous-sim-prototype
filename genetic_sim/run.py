"""
@file run.py
@brief Genetic training: evolve the NN controller by mutating the best car each generation.

This file is only the genetic algorithm: the track schedule, mutation, the
generation loop and the run logs. Racing the cars lives in testing_env.py, the
score in scoring.py and the settings in config.py.

Each generation races a population of n cars on a group of tracks at once (the
races are split over worker processes, see testing_env.race_tracks). A car's
score for the generation is the geometric mean of its scores across the tracks in
the group (scoring.combine_scores). The best car then seeds the next generation:

    car 0        the best car, unchanged (so the best never gets worse)
    cars 1..n-1  copies of the best car with small Gaussian noise on the weights

Tracks: the tracks in start_poses.csv are split into training tracks and held-out
test tracks (config: test_tracks). The training tracks are split into groups of
consecutive tracks. Each group is raced for gens_per_group generations, then the
next group takes over. One pass through all the groups is a cycle, and the run
does `cycles` cycles:

    cycle 1: gens 0..m-1    first group of tracks
             gens m..2m-1   second group
             ...
             TEST           best car races each test track; weights untouched
    cycle 2: same groups again, then TEST again
    ...
    END      the renderer opens and the final best car drives show_track
             (default: the first test track)

Generation 0 is n random nets, or the best car of a saved run (config: load).

Outputs, in runs/genetic_<timestamp>/ (rewritten every generation, so a run
stopped with Ctrl+C keeps everything up to the last finished generation):
    config.json        the settings the run used
    run_info.json      track order, group count, total generations, net size
    history.csv        one row per generation: tracks, best/mean score, per-track
                       score of the best car
    generations.csv    one row per car per generation, with per-track scores
    best.npz           best car of the latest generation, in the weights.npz
                       format, so it can be watched with
                       python testing_env.py --load runs/<run>/best.npz --map Monza
    best_per_gen.npz   params (G, P) best car of every generation, layer_sizes
    population.npz     the whole latest population (weights.npz format)
    test.csv           one row per test track per cycle
    best_cycleC.npz    best car at the end of cycle C

All settings live in config.py (each key is commented there):

    python run.py
"""

import csv
import json
import os
import time
from types import SimpleNamespace

# One BLAS thread per process: the nets are tiny, and with a worker per core
# extra threads only fight each other. Set before numpy loads; workers inherit it.
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_var, '1')

import numpy as np

from config import get_config
from nn_controller import HERE, NNController, load_tracks
from scoring import combine_scores
from testing_env import (TIMESTEP, best_from_saved, choose_workers, make_pool,
                         make_population, race_tracks, show_best, split_jobs,
                         write_population)


def fail(msg):
    """@brief Exit with a config error message.

    @param msg Description of the problem.
    """
    raise SystemExit(f'config error: {msg}')


def check_config(args, all_tracks):
    """@brief Validate the settings.

    @param args       Settings (attribute access).
    @param all_tracks Result of load_tracks().
    @return list Training tracks, in order.
    """
    if args.n < 2:
        fail('n must be at least 2')
    test_tracks = args.test_tracks
    track_names = args.tracks or [t for t in all_tracks if t not in test_tracks]
    unknown = [t for t in track_names + test_tracks if t not in all_tracks]
    if unknown:
        fail(f'unknown track(s) {unknown}; choose from: {", ".join(all_tracks)}')
    if args.show_track not in (None, 'none') and args.show_track not in all_tracks:
        fail(f'unknown show_track {args.show_track!r}')
    overlap = set(track_names) & set(test_tracks)
    if overlap:
        fail(f'track(s) {sorted(overlap)} are both training and test tracks')
    return track_names


def initial_population(args, rng):
    """@brief Generation 0: random nets, or mutants of the best car in a saved run.

    @param args Settings.
    @param rng  numpy Generator for the mutation noise.
    @return tuple (population array (n, P), hidden layer sizes).
    """
    if args.load:
        best_params, hidden = best_from_saved(args.load)
        population = mutate(best_params, args.n, args.sigma, args.mutation_rate, rng)
    else:
        controllers, _ = make_population(args.n, args.seed)
        hidden = tuple(W.shape[1] for W in controllers[0].net.weights[:-1])
        population = np.stack([c.net.get_params() for c in controllers])
    return population, tuple(int(h) for h in hidden)


def mutate(parent, n, sigma, rate, rng):
    """@brief Make n cars from a parent: the unchanged parent plus n-1 mutants.

    @param parent Flat weight vector.
    @param n      Population size.
    @param sigma  Std of the Gaussian noise.
    @param rate   Fraction of weights that get noise.
    @param rng    numpy Generator.
    @return ndarray (n, P), parent in row 0.
    """
    children = np.repeat(parent[None, :], n, axis=0)
    noise = rng.normal(0.0, sigma, size=(n - 1, parent.size))
    if rate < 1.0:
        noise *= rng.random(noise.shape) < rate
    children[1:] += noise
    return children


def track_group(track_names, g, size):
    """@brief The g-th group of `size` consecutive tracks, wrapping round the list.

    @param track_names Training tracks in order.
    @param g           Group index.
    @param size        Tracks per group.
    @return list Track names.
    """
    return [track_names[(g * size + k) % len(track_names)] for k in range(size)]


def init_run(args, track_names, n_groups, total_gens, hidden):
    """@brief Create the run directory and the files that are appended to.

    @param args        Settings.
    @param track_names Training tracks in order.
    @param n_groups    Number of track groups.
    @param total_gens  Total generations.
    @param hidden      Hidden layer sizes.
    @return str Path of the run directory.
    """
    k = args.tracks_per_group
    run_dir = os.path.join(HERE, 'runs', f'genetic_{time.strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(run_dir)
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    with open(os.path.join(run_dir, 'run_info.json'), 'w') as f:
        json.dump({'track_order': track_names, 'groups': n_groups,
                   'total_generations': total_gens, 'hidden': hidden}, f, indent=2)
    with open(os.path.join(run_dir, 'test.csv'), 'w', newline='') as f:
        csv.writer(f).writerow(['cycle', 'gen', 'track', 'score', 'crashed', 'sim_time_s',
                                'distance_m', 'median_speed_mps', 'laps'])
    with open(os.path.join(run_dir, 'history.csv'), 'w', newline='') as f:
        csv.writer(f).writerow(['gen', 'group', 'tracks', 'best_car', 'best_score', 'mean_score',
                                'best_track_scores', 'best_laps', 'best_crashes', 'wall_time_s'])
    with open(os.path.join(run_dir, 'generations.csv'), 'w', newline='') as f:
        csv.writer(f).writerow(['gen', 'car', 'geom_mean'] + [f'track{i}' for i in range(k)] +
                               [f'score{i}' for i in range(k)] + [f'crashed{i}' for i in range(k)])
    return run_dir


def log_generation(run_dir, gen, group, tracks, per_track, scores, geom_mean, best, wall):
    """@brief Print the generation's summary line and append it to history.csv and generations.csv.

    @param run_dir   Run directory.
    @param gen       Generation number.
    @param group     Track group number.
    @param tracks    Tracks raced this generation.
    @param per_track Results from race_tracks().
    @param scores    Per-track scores (tracks, cars).
    @param geom_mean Ranking score per car.
    @param best      Index of the best car.
    @param wall      Wall-clock seconds the generation took.
    """
    best_runs = [per_track[t][best] for t in tracks]
    print(f'gen {gen:4d}  [{", ".join(tracks)}]  best car {best:3d}  geom_mean {geom_mean[best]:8.1f}  '
          f'(mean {geom_mean.mean():7.1f})  per track: '
          + '  '.join(f'{s:7.1f}{"x" if r["crashed"] else " "}'
                      for s, r in zip(scores[:, best], best_runs))
          + f'  {wall:5.1f}s')

    with open(os.path.join(run_dir, 'history.csv'), 'a', newline='') as f:
        csv.writer(f).writerow([
            gen, group, '|'.join(tracks), best, round(geom_mean[best], 3), round(geom_mean.mean(), 3),
            '|'.join(f'{s:.3f}' for s in scores[:, best]),
            '|'.join(str(r['laps']) for r in best_runs),
            '|'.join(str(int(r['crashed'])) for r in best_runs),
            round(wall, 2)])
    with open(os.path.join(run_dir, 'generations.csv'), 'a', newline='') as f:
        w = csv.writer(f)
        for car in range(len(geom_mean)):
            w.writerow([gen, car, round(geom_mean[car], 3)] + tracks +
                       [round(s, 3) for s in scores[:, car]] +
                       [int(per_track[t][car]['crashed']) for t in tracks])


def save_generation(run_dir, population, best_params, best_history, hidden):
    """@brief Write the latest population, best car and best-per-generation history.

    @param run_dir      Run directory.
    @param population   Array (n, P) of flat weight vectors.
    @param best_params  Flat weights of the best car.
    @param best_history Best car of every generation so far.
    @param hidden       Hidden layer sizes.
    """
    net = NNController(hidden=hidden).net
    layer_sizes = np.array([net.weights[0].shape[0]] + [W.shape[1] for W in net.weights])
    write_population(os.path.join(run_dir, 'population.npz'), population, hidden)
    write_population(os.path.join(run_dir, 'best.npz'), best_params[None, :], hidden)
    np.savez(os.path.join(run_dir, 'best_per_gen.npz'), params=np.stack(best_history),
             layer_sizes=layer_sizes)


def run_test(pool, cache_key, cycle, gen, params, hidden, test_tracks, all_tracks, max_steps, run_dir):
    """@brief Race the best car on every test track. Results are logged only; no weights change.

    @param pool         Worker pool.
    @param cache_key    Env cache key for the workers.
    @param cycle        Cycle index (0-based).
    @param gen          Generation the best car came from.
    @param params       Flat weights of the best car.
    @param hidden       Hidden layer sizes.
    @param test_tracks  Held-out tracks.
    @param all_tracks   Result of load_tracks().
    @param max_steps    Step limit per race.
    @param run_dir      Run directory.
    """
    write_population(os.path.join(run_dir, f'best_cycle{cycle + 1}.npz'), params[None, :], hidden)
    if not test_tracks:
        return
    per_track = race_tracks(pool, cache_key, test_tracks, all_tracks, params[None, :], hidden,
                            max_steps, workers=len(test_tracks))
    print(f'----- TEST after cycle {cycle + 1} (best car of gen {gen}, weights not updated)')
    with open(os.path.join(run_dir, 'test.csv'), 'a', newline='') as f:
        w = csv.writer(f)
        for t in test_tracks:
            r = per_track[t][0]
            print(f'      {t:14s} score {r["score"]:8.1f}  {"crashed" if r["crashed"] else "survived":8s}  '
                  f't={r["sim_time_s"]:6.2f}s  dist={r["distance_m"]:7.1f}m  laps={r["laps"]}')
            w.writerow([cycle + 1, gen, t, r['score'], int(r['crashed']), r['sim_time_s'],
                        r['distance_m'], r['median_speed_mps'], r['laps']])


def main():
    """@brief Run the genetic training described in the module header."""
    args = SimpleNamespace(**get_config())
    all_tracks = load_tracks()
    test_tracks = args.test_tracks
    track_names = check_config(args, all_tracks)
    rng = np.random.default_rng(args.seed)
    if args.shuffle_tracks:
        track_names = [str(t) for t in rng.permutation(track_names)]

    k = args.tracks_per_group
    n_groups = -(-len(track_names) // k)  # ceil; a short last group wraps round to the start
    if len(track_names) % k:
        print(f'Note: {len(track_names)} training tracks do not split evenly into groups of {k}; '
              f'the last group wraps round to the first tracks.')
    gens_per_cycle = n_groups * args.gens_per_group
    total_gens = args.cycles * gens_per_cycle
    max_steps = int(args.max_time / TIMESTEP)

    population, hidden = initial_population(args, rng)
    run_dir = init_run(args, track_names, n_groups, total_gens, hidden)
    best_history = []
    print(f'Run: {run_dir}\n{args.n} cars, {len(track_names)} training tracks in {n_groups} groups of {k}, '
          f'{args.gens_per_group} gens per group, {args.cycles} cycles = {total_gens} generations\n'
          f'Test tracks: {", ".join(test_tracks) or "none"}\n'
          f'sigma={args.sigma}, mutation rate={args.mutation_rate}\n')

    workers = choose_workers(args.workers, args.n * k)
    slices = split_jobs(args.n, k, workers)
    print(f'{workers} worker processes: {args.n} cars x {k} tracks = {args.n * k} car-races per generation, '
          f'as {len(slices) * k} jobs of up to {max(hi - lo for lo, hi in slices)} cars\n')
    pool = make_pool(workers)
    gen = 0
    try:
        while gen < total_gens:
            group = gen // args.gens_per_group  # counts up across cycles, so env caches reset per group
            cycle = gen // gens_per_cycle
            tracks = track_group(track_names, group % n_groups, k)
            if gen % gens_per_cycle == 0:
                print(f'===== cycle {cycle + 1}/{args.cycles}')
            start = time.time()
            per_track = race_tracks(pool, group, tracks, all_tracks, population, hidden, max_steps, workers)

            scores, geom_mean = combine_scores(per_track, tracks)
            best = int(np.argmax(geom_mean))
            best_params = population[best].copy()
            best_history.append(best_params)

            log_generation(run_dir, gen, group, tracks, per_track, scores, geom_mean, best,
                           time.time() - start)
            save_generation(run_dir, population, best_params, best_history, hidden)

            if (gen + 1) % gens_per_cycle == 0:
                run_test(pool, ('test', cycle), cycle, gen, best_params, hidden, test_tracks,
                         all_tracks, max_steps, run_dir)
            elif (gen + 1) % args.gens_per_group == 0:
                print(f'--- group {group % n_groups} done; next: '
                      f'{", ".join(track_group(track_names, (group + 1) % n_groups, k))}')
            population = mutate(best_params, args.n, args.sigma, args.mutation_rate, rng)
            gen += 1
    except KeyboardInterrupt:
        print('\nStopped.')
        pool.terminate()
        pool.join()
        print(f'{gen} generations saved to {run_dir}')
        return
    pool.close()
    pool.join()
    print(f'{gen} generations saved to {run_dir}')

    if args.show_track != 'none':
        show_track = args.show_track or (test_tracks[0] if test_tracks else track_names[0])
        show_best(best_params, hidden, show_track, all_tracks, max_steps)


if __name__ == '__main__':
    main()
