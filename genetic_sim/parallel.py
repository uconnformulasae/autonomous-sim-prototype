"""
@file parallel.py
@brief Race populations across worker processes. Knows nothing about training or configs.

A Racer owns a pool of worker processes. Give it any population (flat weight
vectors), any tracks and a step limit, and it returns each car's result on each
track. The cars are ghosts to each other (see testing_env.make_ghost_env), so
splitting them across processes does not change any car's result, only how many
cores share the work.

    with Racer(workers=0, n_races=500 * 2) as racer:
        per_track = racer.race('key', ['Monza', 'Spa'], all_tracks, params, hidden, max_steps)

The same Racer can be reused for any number of races with different settings.
"""

import os
import warnings
from multiprocessing import Pool

import numpy as np

from nn_controller import NNController
from testing_env import make_ghost_env, race

# Each worker process keeps its envs between calls so a track's map is only
# loaded once per key. The cache is dropped when the key changes.
_envs = {}
_envs_key = None


def _quiet_worker():
    """@brief Pool initializer: silence the repeated RK4 integrator warning."""
    # Every env build warns that RK4 is the integrator; once is plenty.
    warnings.filterwarnings('ignore', message='Chosen integrator is RK4')


def _race_track(job):
    """@brief Race a chunk of the population on one track. Runs in a worker process.

    @param job Tuple (key, track, lo, map_path, start_pose, params, hidden, max_steps).
    @return tuple (track, lo, results).
    """
    global _envs_key
    key, track, lo, map_path, start_pose, params, hidden, max_steps = job
    if key != _envs_key:
        _envs.clear()
        _envs_key = key
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


class Racer:
    """@brief A pool of worker processes that races populations on tracks."""

    def __init__(self, workers, max_races):
        """@brief Start the worker processes.

        @param workers   Requested worker count (0: 75% of the CPU threads).
        @param max_races Most car-races a single race() call will have (cars x tracks);
                         more workers than this would only sit idle.
        """
        default = max(1, (os.cpu_count() or 1) * 3 // 4)
        ## Number of worker processes.
        self.workers = min(workers or default, max_races)
        self._pool = Pool(self.workers, initializer=_quiet_worker)

    def describe(self, n_cars, n_tracks):
        """@brief One-line summary of how a race of n_cars x n_tracks is shared out.

        @param n_cars   Population size.
        @param n_tracks Tracks raced at once.
        @return str
        """
        workers = min(self.workers, n_cars * n_tracks)
        slices = split_jobs(n_cars, n_tracks, workers)
        return (f'{workers} worker processes: {n_cars} cars x {n_tracks} tracks = {n_cars * n_tracks} '
                f'car-races per generation, as {len(slices) * n_tracks} jobs of up to '
                f'{max(hi - lo for lo, hi in slices)} cars')

    def race(self, key, tracks, all_tracks, params, hidden, max_steps):
        """@brief Race every car on every track.

        @param key        Worker env caches reset when this changes, so give each
                          group of races that share tracks one key.
        @param tracks     Track names to race.
        @param all_tracks Result of load_tracks().
        @param params     Array (cars, P) of flat weight vectors.
        @param hidden     Hidden layer sizes.
        @param max_steps  Step limit per race.
        @return dict {track: [result dict per car]}.
        """
        workers = min(self.workers, len(params) * len(tracks))
        jobs = [(key, t, lo, *all_tracks[t], params[lo:hi], hidden, max_steps)
                for t in tracks for lo, hi in split_jobs(len(params), len(tracks), workers)]
        per_track = {t: [None] * len(params) for t in tracks}
        for t, lo, results in self._pool.imap_unordered(_race_track, jobs):
            per_track[t][lo:lo + len(results)] = results
        return per_track

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        """@brief Close the pool; terminate it instead if leaving on an exception (e.g. Ctrl+C)."""
        if exc_type is None:
            self._pool.close()
        else:
            self._pool.terminate()
        self._pool.join()
