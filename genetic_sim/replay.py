"""
@file replay.py
@brief Replay a saved set of weights (.npz) on the test tracks, with visuals and scores.

Give it a weights file written by training (best.npz, best_cycleC.npz,
population.npz, a grid search's best.npz, or testing_env's weights.npz). The car
drives each track in turn in the render window, a score table is printed at the
end, and the table is saved as a csv next to the weights.

Tracks, unless you pick them:
    the test_tracks from the config.json in the same run folder, else from
    config.get_config(). Add --val for the validation tracks, or --all for every
    track.

If the file holds several cars (population.npz), the best one is used (by
results.csv next to it if there is one, else the first); pick one with --car.

    python replay.py runs/genetic_20261122_155905/best.npz
    python replay.py runs/<run>/best_cycle2.npz --val
    python replay.py runs/<run>/best.npz --tracks Monza Spa --max-time 90
    python replay.py runs/<run>/best.npz --headless        # scores only, much faster

Closing the window ends the visuals; the remaining tracks are still scored,
without a window. Between tracks the finished race stays up for --hold seconds.
"""

import argparse
import csv
import json
import os
import time
import warnings

import numpy as np

from config import get_config
from nn_controller import NNController, load_tracks
from scoring import SCORE_FLOOR
from testing_env import (TIMESTEP, add_race_view, best_from_saved, load_population,
                         make_ghost_env, race, is_window_closed)
from f110_gym.envs.f110_env import F110Env


def resolve_params(path, car):
    """@brief Load the flat weights of the car to replay.

    @param path Weights .npz.
    @param car  Row to use, or None for the best car (see best_from_saved()).
    @return tuple (flat params, hidden layer sizes).
    """
    if car is None:
        return best_from_saved(path)
    controllers, _ = load_population(path)
    if not 0 <= car < len(controllers):
        raise SystemExit(f'--car {car} out of range; the file holds {len(controllers)} car(s)')
    hidden = tuple(int(h) for h in np.load(path)['layer_sizes'][1:-1])
    return controllers[car].net.get_params(), hidden


def default_tracks(path, include_val, all_tracks):
    """@brief The test tracks (and validation tracks if asked) of the run the weights came from.

    @param path        Weights .npz.
    @param include_val Whether to add the validation tracks.
    @param all_tracks  Result of load_tracks().
    @return list Track names.
    """
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(path)), 'config.json')
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
    else:
        cfg = get_config()
    tracks = list(cfg.get('test_tracks') or [])
    if include_val:
        tracks += [t for t in (cfg.get('val_tracks') or []) if t not in tracks]
    return tracks


def close_window():
    """@brief Close the render window and forget it so the next track opens a fresh one."""
    renderer = F110Env.renderer
    if renderer is not None:
        try:
            import pyglet
            pyglet.window.Window.close(renderer)  # EnvRenderer.on_close would raise
        except Exception:
            pass
    F110Env.renderer = None
    F110Env.render_callbacks = []


def replay_track(params, hidden, track, all_tracks, max_steps, render, hold):
    """@brief Drive the car once on one track.

    @param params    Flat weights.
    @param hidden    Hidden layer sizes.
    @param track     Track name.
    @param all_tracks Result of load_tracks().
    @param max_steps Step limit.
    @param render    Whether to show the window.
    @param hold      Seconds to keep the finished race on screen.
    @return tuple (result dict, window_still_open).
    """
    c = NNController(hidden=hidden)
    c.net.set_params(params)
    map_path, start_pose = all_tracks[track]
    env = make_ghost_env(map_path, 1)
    if render:
        add_race_view(env)
    (r,), window_open = race(env, [c], start_pose, max_steps, render)
    if render and not window_open:
        # Window closed mid-race: the result is partial, so score the track again without it.
        close_window()
        env = make_ghost_env(map_path, 1)
        (r,), _ = race(env, [c], start_pose, max_steps, render=False)
    elif render:
        end = time.time() + hold
        try:
            while time.time() < end:
                env.render(mode='human')
        except Exception as ex:
            if not is_window_closed(ex):
                raise
            window_open = False
        if window_open:
            close_window()
    return r, window_open


def main():
    """@brief Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('weights', help='weights .npz (best.npz, population.npz, ...)')
    parser.add_argument('--car', type=int, help='row of the file to replay (default: the best car)')
    parser.add_argument('--tracks', nargs='+', help='tracks to race (default: the run\'s test tracks)')
    parser.add_argument('--val', action='store_true', help='also race the run\'s validation tracks')
    parser.add_argument('--all', action='store_true', help='race every track')
    parser.add_argument('--max-time', type=float, default=None,
                        help='sim seconds per track (default: the run\'s max_time, else 60)')
    parser.add_argument('--hold', type=float, default=2.0, help='seconds to keep each finished race on screen')
    parser.add_argument('--headless', action='store_true', help='no window, scores only')
    parser.add_argument('--no-save', action='store_true', help='do not write the results csv')
    args = parser.parse_args()
    warnings.filterwarnings('ignore', message='Chosen integrator is RK4')

    all_tracks = load_tracks()
    if args.all:
        tracks = list(all_tracks)
    else:
        tracks = args.tracks or default_tracks(args.weights, args.val, all_tracks)
    unknown = [t for t in tracks if t not in all_tracks]
    if unknown:
        parser.error(f'unknown track(s) {unknown}; choose from: {", ".join(all_tracks)}')
    if not tracks:
        parser.error('no tracks to race; pass --tracks, --val or --all')

    max_time = args.max_time
    if max_time is None:
        cfg_path = os.path.join(os.path.dirname(os.path.abspath(args.weights)), 'config.json')
        max_time = 60.0
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                max_time = float(json.load(f).get('max_time', max_time))
    max_steps = int(max_time / TIMESTEP)

    params, hidden = resolve_params(args.weights, args.car)
    print(f'Replaying {args.weights} (hidden {hidden}) on {len(tracks)} track(s), {max_time:g}s each\n')

    render = not args.headless
    rows = []
    for t in tracks:
        print(f'{t}: driving...', flush=True)
        r, window_open = replay_track(params, hidden, t, all_tracks, max_steps, render, args.hold)
        render = render and window_open
        rows.append({'track': t, **r})
        print(f'  score {r["score"]:8.1f}  {"crashed" if r["crashed"] else "survived":8s}  '
              f't={r["sim_time_s"]:6.2f}s  dist={r["distance_m"]:7.1f}m  '
              f'median v={r["median_speed_mps"]:5.2f}m/s  laps={r["laps"]}')

    scores = np.array([max(r['score'], SCORE_FLOOR) for r in rows])
    print('\n' + '=' * 78)
    print(f'{"track":16s}{"score":>10s}  {"result":9s}{"time s":>8s}{"dist m":>9s}{"laps":>6s}')
    for r in rows:
        print(f'{r["track"]:16s}{r["score"]:10.1f}  {"crashed" if r["crashed"] else "survived":9s}'
              f'{r["sim_time_s"]:8.2f}{r["distance_m"]:9.1f}{r["laps"]:6d}')
    geom = float(np.exp(np.mean(np.log(scores))))
    print(f'{"mean":16s}{np.mean([r["score"] for r in rows]):10.1f}')
    print(f'{"geometric mean":16s}{geom:10.1f}   crashes: {sum(r["crashed"] for r in rows)}/{len(rows)}')

    if not args.no_save:
        out = os.path.join(os.path.dirname(os.path.abspath(args.weights)),
                           f'replay_{os.path.splitext(os.path.basename(args.weights))[0]}_'
                           f'{time.strftime("%Y%m%d_%H%M%S")}.csv')
        with open(out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f'\nSaved {out}')


if __name__ == '__main__':
    main()
