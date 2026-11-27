"""
@file run.py
@brief Genetic training: evolve the NN controller by mutating the best car each generation.

This file is the genetic algorithm and the run logs: the track schedule, mutation,
the generation loop, and the grid search that repeats it for several configs.
Racing the cars in parallel lives in parallel.py, the race itself in
testing_env.py, the score in scoring.py and the settings in config.py.

Each generation races a population of n cars on a group of tracks at once (the
races are split over worker processes, see parallel.py). A car's
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

    python run.py            train once with get_config()
    python run.py --resume runs/<run>/best.npz
                             continue from saved weights with that run's exact settings
                             (read from the config.json / best_config.json beside the
                             file). Saved weights are the parents: a file with exactly n
                             cars (population.npz) is generation 0 as it is, anything
                             else (best.npz, best_cycleC.npz) is mutated into n cars.
                             --load PATH in config.py does the same with the current config.
    python run.py --grid     grid search: train once per combination in get_grid(),
                             score each final best car on val_tracks (never trained
                             on), and save the best combination to
                             runs/gridsearch_<timestamp>/ (grid_search.log,
                             best_config.json, best.npz)
"""

import argparse
import csv
import itertools
import json
import os
import time
from types import SimpleNamespace

# One BLAS thread per process: the nets are tiny, and with a worker per core
# extra threads only fight each other. Set before numpy loads; workers inherit it.
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_var, '1')

import numpy as np

from config import get_config, get_grid
from nn_controller import HERE, NNController, load_tracks
from parallel import Racer
from scoring import combine_scores
from testing_env import (TIMESTEP, best_from_saved, load_population, make_population,
                         show_best, write_population)


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
    if args.cycles < 1:
        fail('cycles must be at least 1')
    test_tracks = args.test_tracks
    val_tracks = getattr(args, 'val_tracks', None) or []
    held_out = test_tracks + val_tracks
    track_names = args.tracks or [t for t in all_tracks if t not in held_out]
    unknown = [t for t in track_names + held_out if t not in all_tracks]
    if unknown:
        fail(f'unknown track(s) {unknown}; choose from: {", ".join(all_tracks)}')
    if args.show_track not in (None, 'none') and args.show_track not in all_tracks:
        fail(f'unknown show_track {args.show_track!r}')
    overlap = set(track_names) & set(held_out)
    if overlap:
        fail(f'track(s) {sorted(overlap)} are both training and test/validation tracks')
    if set(test_tracks) & set(val_tracks):
        fail(f'track(s) {sorted(set(test_tracks) & set(val_tracks))} are both test and validation tracks')
    return track_names


def initial_population(args, rng):
    """@brief Generation 0: random nets, or mutants of the best car in a saved run.

    @param args Settings.
    @param rng  numpy Generator for the mutation noise.
    @note A saved file with exactly args.n cars is used as generation 0 directly;
          otherwise its best car is mutated into n cars.
    @return tuple (population array (n, P), hidden layer sizes).
    """
    if args.load:
        saved, _ = load_population(args.load)
        if len(saved) == args.n:  # a whole saved population: carry on from it unchanged
            population = np.stack([c.net.get_params() for c in saved])
            hidden = tuple(W.shape[1] for W in saved[0].net.weights[:-1])
        else:
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


def training_order(args, track_names, rng):
    """@brief The training tracks in the order they are raced (shuffled once if configured).

    @param args        Settings.
    @param track_names Training tracks.
    @param rng         numpy Generator (only used when shuffling).
    @return list Track names.
    """
    if args.shuffle_tracks:
        return [str(t) for t in rng.permutation(track_names)]
    return track_names


def schedule(args, n_tracks):
    """@brief Size of the training schedule.

    @param args     Settings.
    @param n_tracks Number of training tracks.
    @return tuple (groups, generations per cycle, total generations).
    """
    n_groups = -(-n_tracks // args.tracks_per_group)  # ceil; a short last group wraps round to the start
    gens_per_cycle = n_groups * args.gens_per_group
    return n_groups, gens_per_cycle, args.cycles * gens_per_cycle


def evolve(racer, args, track_names, all_tracks, population, hidden, rng, key=0):
    """@brief The genetic loop: race, pick the best car, mutate it, repeat.

    Yields once per generation, after the race and before the next population
    is made, so the caller can log or save.

    @param racer       parallel.Racer that does the racing.
    @param args        Settings.
    @param track_names Training tracks in the order to race them.
    @param all_tracks  Result of load_tracks().
    @param population  Generation 0, array (n, P).
    @param hidden      Hidden layer sizes.
    @param rng         numpy Generator for the mutation noise.
    @param key         Distinguishes this run's worker env caches from other runs'.
    @yield SimpleNamespace with gen, group, cycle, tracks, per_track, scores,
           geom_mean, best (index), best_params, population (the one just raced)
           and wall (seconds).
    """
    k = args.tracks_per_group
    n_groups, gens_per_cycle, total_gens = schedule(args, len(track_names))
    max_steps = int(args.max_time / TIMESTEP)
    for gen in range(total_gens):
        group = gen // args.gens_per_group  # counts up across cycles, so env caches reset per group
        tracks = track_group(track_names, group % n_groups, k)
        start = time.time()
        per_track = racer.race((key, group), tracks, all_tracks, population, hidden, max_steps)
        scores, geom_mean = combine_scores(per_track, tracks, args.score_metric)
        best = int(np.argmax(geom_mean))
        best_params = population[best].copy()
        yield SimpleNamespace(gen=gen, group=group, cycle=gen // gens_per_cycle, tracks=tracks,
                              per_track=per_track, scores=scores, geom_mean=geom_mean,
                              best=best, best_params=best_params, population=population,
                              wall=time.time() - start)
        population = mutate(best_params, args.n, args.sigma, args.mutation_rate, rng)


def init_run(args, track_names, n_groups, total_gens, hidden):
    """@brief Create the run directory and the files that are appended to.

    @param args        Settings.
    @param track_names Training tracks in order.
    @param n_groups    Number of track groups.
    @param total_gens  Total generations.
    @param hidden      Hidden layer sizes.
    @return str Path of the run directory.
    """
    base = os.path.join(HERE, 'runs', f'genetic_{time.strftime("%Y%m%d_%H%M%S")}')
    run_dir, suffix = base, 0
    while True:  # two runs started in the same second must not share a folder
        try:
            os.makedirs(run_dir)
            break
        except FileExistsError:
            suffix += 1
            run_dir = f'{base}_{suffix}'
    write_run_files(run_dir, args, {'track_order': track_names, 'groups': n_groups,
                                    'total_generations': total_gens, 'hidden': hidden})
    return run_dir


def csv_headers(k):
    """@brief Header row of each CSV in the run directory.

    @param k Tracks per group.
    @return dict {file name: header list}.
    """
    return {
        'test.csv': ['cycle', 'gen', 'track', 'score', 'crashed', 'sim_time_s',
                     'distance_m', 'median_speed_mps', 'laps'],
        'history.csv': ['gen', 'group', 'tracks', 'best_car', 'best_score', 'mean_score',
                        'best_track_scores', 'best_laps', 'best_crashes', 'wall_time_s'],
        'generations.csv': ['gen', 'car', 'geom_mean'] + [f'track{i}' for i in range(k)] +
                           [f'score{i}' for i in range(k)] + [f'crashed{i}' for i in range(k)],
    }


def write_run_files(run_dir, args, run_info):
    """@brief Write config.json, run_info.json and the CSV headers into run_dir.

    @param run_dir  Run directory (must exist).
    @param args     Settings.
    @param run_info Contents of run_info.json.
    """
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    with open(os.path.join(run_dir, 'run_info.json'), 'w') as f:
        json.dump(run_info, f, indent=2)
    for name, header in csv_headers(args.tracks_per_group).items():
        with open(os.path.join(run_dir, name), 'w', newline='') as f:
            csv.writer(f).writerow(header)


def ensure_run_dir(run_dir, args, run_info):
    """@brief Recreate the run directory (and its config and CSV headers) if it was deleted mid-run.

    @param run_dir  Run directory.
    @param args     Settings.
    @param run_info Contents of run_info.json.
    """
    if not os.path.isdir(run_dir):
        print(f'WARNING: run directory {run_dir} disappeared; recreating it. '
              f'Logs from before this generation are lost.')
        os.makedirs(run_dir, exist_ok=True)
        write_run_files(run_dir, args, run_info)


def append_csv(run_dir, name, k, rows):
    """@brief Append rows to a CSV, writing its header first if the file is missing.

    @param run_dir Run directory.
    @param name    File name, a key of csv_headers().
    @param k       Tracks per group.
    @param rows    Iterable of row lists.
    """
    path = os.path.join(run_dir, name)
    new = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        w = csv.writer(f)
        if new:
            w.writerow(csv_headers(k)[name])
        w.writerows(rows)


def log_generation(run_dir, gen, group, tracks, per_track, scores, geom_mean, best, wall):
    """@brief Print the generation's summary line and append it to history.csv and generations.csv.

    @param run_dir   Run directory.
    @param gen       Generation number.
    @param group     Track group number.
    @param tracks    Tracks raced this generation.
    @param per_track Results from Racer.race().
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

    k = len(tracks)
    append_csv(run_dir, 'history.csv', k, [[
        gen, group, '|'.join(tracks), best, round(geom_mean[best], 3), round(geom_mean.mean(), 3),
        '|'.join(f'{s:.3f}' for s in scores[:, best]),
        '|'.join(str(r['laps']) for r in best_runs),
        '|'.join(str(int(r['crashed'])) for r in best_runs),
        round(wall, 2)]])
    append_csv(run_dir, 'generations.csv', k, (
        [gen, car, round(geom_mean[car], 3)] + tracks +
        [round(s, 3) for s in scores[:, car]] +
        [int(per_track[t][car]['crashed']) for t in tracks]
        for car in range(len(geom_mean))))


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
    path = os.path.join(run_dir, 'best_per_gen.npz')
    tmp = path[:-4] + '.tmp.npz'
    np.savez(tmp, params=np.stack(best_history), layer_sizes=layer_sizes,
             seeds=np.arange(len(best_history)))
    os.replace(tmp, path)


def run_test(racer, cache_key, cycle, gen, params, hidden, test_tracks, all_tracks, max_steps, run_dir):
    """@brief Race the best car on every test track. Results are logged only; no weights change.

    @param racer        parallel.Racer.
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
    per_track = racer.race(cache_key, test_tracks, all_tracks, params[None, :], hidden, max_steps)
    print(f'----- TEST after cycle {cycle + 1} (best car of gen {gen}, weights not updated)')
    rows = []
    for t in test_tracks:
        r = per_track[t][0]
        print(f'      {t:14s} score {r["score"]:8.1f}  {"crashed" if r["crashed"] else "survived":8s}  '
              f't={r["sim_time_s"]:6.2f}s  dist={r["distance_m"]:7.1f}m  laps={r["laps"]}')
        rows.append([cycle + 1, gen, t, r['score'], int(r['crashed']), r['sim_time_s'],
                     r['distance_m'], r['median_speed_mps'], r['laps']])
    append_csv(run_dir, 'test.csv', 0, rows)


def train(racer, args, track_names, all_tracks, key=0, log=False):
    """@brief One full genetic training run: build generation 0, evolve it, return the best car.

    @param racer       parallel.Racer.
    @param args        Settings (attribute access).
    @param track_names Training tracks (from check_config()).
    @param all_tracks  Result of load_tracks().
    @param key         Keeps this run's worker env caches apart from other runs'.
    @param log         True to print progress, write the run directory and race the
                       test tracks after every cycle. False is a quiet run that
                       writes nothing (used by the grid search).
    @return SimpleNamespace with best_params (flat weights), hidden, gens (generations
            run) and run_dir (None unless log).
    """
    rng = np.random.default_rng(args.seed)
    track_names = training_order(args, track_names, rng)
    k = args.tracks_per_group
    n_groups, gens_per_cycle, total_gens = schedule(args, len(track_names))
    max_steps = int(args.max_time / TIMESTEP)
    population, hidden = initial_population(args, rng)

    run_dir = None
    best_history = []
    run_info = {'track_order': track_names, 'groups': n_groups,
                'total_generations': total_gens, 'hidden': hidden}
    if log:
        run_dir = init_run(args, track_names, n_groups, total_gens, hidden)
        if len(track_names) % k:
            print(f'Note: {len(track_names)} training tracks do not split evenly into groups of {k}; '
                  f'the last group wraps round to the first tracks.')
        print(f'Run: {run_dir}\n{args.n} cars, {len(track_names)} training tracks in {n_groups} groups of {k}, '
              f'{args.gens_per_group} gens per group, {args.cycles} cycles = {total_gens} generations\n'
              f'Test tracks: {", ".join(args.test_tracks) or "none"}\n'
              f'sigma={args.sigma}, mutation rate={args.mutation_rate}\n\n'
              f'{racer.describe(args.n, k)}\n')

    for g in evolve(racer, args, track_names, all_tracks, population, hidden, rng, key):
        if log:
            if g.gen % gens_per_cycle == 0:
                print(f'===== cycle {g.cycle + 1}/{args.cycles}')
            best_history.append(g.best_params)
            ensure_run_dir(run_dir, args, run_info)
            # weights first: a logging failure must never cost the trained car
            save_generation(run_dir, g.population, g.best_params, best_history, hidden)
            log_generation(run_dir, g.gen, g.group, g.tracks, g.per_track, g.scores, g.geom_mean,
                           g.best, g.wall)
            if (g.gen + 1) % gens_per_cycle == 0:
                run_test(racer, ('test', g.cycle), g.cycle, g.gen, g.best_params, hidden,
                         args.test_tracks, all_tracks, max_steps, run_dir)
            elif (g.gen + 1) % args.gens_per_group == 0:
                print(f'--- group {g.group % n_groups} done; next: '
                      f'{", ".join(track_group(track_names, (g.group + 1) % n_groups, k))}')
    return SimpleNamespace(best_params=g.best_params, hidden=hidden, gens=g.gen + 1, run_dir=run_dir)


def grid_combos(grid, base):
    """@brief Every combination of the grid values.

    @param grid dict {setting: list of values} from get_grid().
    @param base dict of the default settings from get_config().
    @return list of dicts {setting: value}, one per combination.
    """
    bad = [k for k, v in grid.items() if k not in base or not v]
    if bad:
        fail(f'grid setting(s) {bad} are not in get_config() or have no values')
    return [dict(zip(grid, values)) for values in itertools.product(*grid.values())]


def validate(racer, base, params, hidden, all_tracks, key):
    """@brief Score a car on the validation tracks, always with the base config's max_time.

    @param racer      parallel.Racer.
    @param base       Default settings from get_config().
    @param params     Flat weights of the car.
    @param hidden     Hidden layer sizes.
    @param all_tracks Result of load_tracks().
    @param key        Env cache key.
    @return tuple (geometric-mean score, per-track result dicts).
    """
    val = base['val_tracks']
    per_track = racer.race(key, val, all_tracks, params[None, :], hidden,
                           int(base['max_time'] / TIMESTEP))
    _, geom_mean = combine_scores(per_track, val, base.get('score_metric', 'score'))
    return float(geom_mean[0]), [per_track[t][0] for t in val]


def grid_search(racer, base, grid, track_names, all_tracks):
    """@brief Train once per combination of settings and keep the one that validates best.

    Writes grid_search.log (a line per combination, then the ideal settings),
    best_config.json and best.npz to runs/gridsearch_<timestamp>/.

    @param racer       parallel.Racer.
    @param base        Default settings from get_config().
    @param grid        {setting: list of values} from get_grid().
    @param track_names Training tracks.
    @param all_tracks  Result of load_tracks().
    """
    combos = grid_combos(grid, base)
    run_dir = os.path.join(HERE, 'runs', f'gridsearch_{time.strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(run_dir)

    def log(msg):
        print(msg)
        with open(os.path.join(run_dir, 'grid_search.log'), 'a') as f:
            f.write(msg + '\n')

    log(f'{len(combos)} combinations of {grid}\n'
        f'train: {", ".join(track_names)}\nvalidate: {", ".join(base["val_tracks"])}\n')
    best_score = None
    for i, combo in enumerate(combos):
        start = time.time()
        trained = train(racer, SimpleNamespace(**{**base, **combo}), track_names, all_tracks, key=i)
        score, results = validate(racer, base, trained.best_params, trained.hidden, all_tracks, ('val', i))
        per_track = '  '.join(f'{t}:{r["score"]:.1f}{"x" if r["crashed"] else ""}'
                              for t, r in zip(base['val_tracks'], results))
        log(f'[{i + 1}/{len(combos)}] {combo}  val {score:.1f}  {per_track}  ({time.time() - start:.0f}s)')
        if best_score is None or score > best_score:
            best_score, best_combo = score, combo
            with open(os.path.join(run_dir, 'best_config.json'), 'w') as f:
                json.dump({**base, **combo, 'val_score': score}, f, indent=2)
            write_population(os.path.join(run_dir, 'best.npz'), trained.best_params[None, :], trained.hidden)
    log(f'\nIDEAL SETTINGS (val {best_score:.1f}): '
        + ', '.join(f'{k}={best_combo[k]!r}' for k in grid) + f'\nSaved to {run_dir}')


def saved_config(weights):
    """@brief The settings a saved weights file was trained with.

    @param weights Path to an .npz from a run or grid search folder.
    @return dict Settings from config.json or best_config.json beside the file.
    """
    folder = os.path.dirname(os.path.abspath(weights))
    for name in ('config.json', 'best_config.json'):
        path = os.path.join(folder, name)
        if os.path.exists(path):
            with open(path) as f:
                cfg = json.load(f)
            cfg.pop('val_score', None)
            return cfg
    fail(f'no config.json or best_config.json next to {weights}')


def main():
    """@brief Train with the config, or with --grid search the config grid."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--grid', action='store_true',
                        help='try every combination in config.get_grid() and keep the one that validates best')
    parser.add_argument('--resume', metavar='WEIGHTS',
                        help='start from saved weights (.npz) as the parents, using the exact '
                             'settings of the run they came from')
    cli = parser.parse_args()
    use_grid = cli.grid
    if use_grid and cli.resume:
        fail('--resume and --grid cannot be combined')

    base = get_config()
    if cli.resume:
        if not os.path.exists(cli.resume):
            fail(f'--resume file not found: {cli.resume}')
        base = {**base, **saved_config(cli.resume), 'load': os.path.abspath(cli.resume)}
        print(f'Resuming from {cli.resume} with the settings of the run it came from (n={base["n"]}, '
              f'sigma={base["sigma"]}, mutation_rate={base["mutation_rate"]}, seed={base["seed"]})')
    args = SimpleNamespace(**base)
    all_tracks = load_tracks()
    track_names = check_config(args, all_tracks)
    if use_grid:
        if not base['val_tracks']:
            fail('val_tracks is empty; the grid search needs validation tracks')
        grid = get_grid()
        max_races = max(c.get('n', args.n) * c.get('tracks_per_group', args.tracks_per_group)
                        for c in grid_combos(grid, base))
    else:
        max_races = args.n * args.tracks_per_group

    try:
        with Racer(args.workers, max_races) as racer:
            if use_grid:
                grid_search(racer, base, grid, track_names, all_tracks)
                return
            result = train(racer, args, track_names, all_tracks, log=True)
    except KeyboardInterrupt:
        print('\nStopped. Everything up to the last finished generation is saved in the run directory.')
        return
    print(f'{result.gens} generations saved to {result.run_dir}')

    if args.show_track != 'none':
        show_track = args.show_track or (args.test_tracks[0] if args.test_tracks else track_names[0])
        show_best(result.best_params, result.hidden, show_track, all_tracks,
                  int(args.max_time / TIMESTEP))


if __name__ == '__main__':
    main()
