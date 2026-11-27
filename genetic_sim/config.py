"""@file config.py
@brief Settings for run.py. Edit the values here, then run `python run.py`.
"""


def get_config():
    """@brief Build the settings dictionary used by run.py.

    @return dict Settings keyed by name:
        - n: cars per generation
        - gens_per_group: generations on each track group before moving on
        - tracks_per_group: tracks raced in parallel each generation
        - cycles: passes through all the training groups
        - sigma: std of the Gaussian noise added to mutated weights
        - mutation_rate: fraction of weights that get noise in each mutant
        - max_time: sim seconds per race
        - seed: seed for gen-0 nets and mutation noise
        - tracks: training tracks in order (None: every non-test track)
        - test_tracks: held-out tracks raced by the best car after every cycle
        - val_tracks: validation tracks, used only by `python run.py --grid` to compare settings
        - score_metric: 'score' (general driving) or 'race_score' (pure speed, crash penalty)
        - shuffle_tracks: shuffle the training track order once at the start
        - load: path to a weights.npz / best.npz to start from (None: random)
        - show_track: track to render the final best car on
        - workers: worker processes (0: 75% of CPU threads)
    """
    CONFIG = {
        'n': 500,                # cars per generation
        'gens_per_group': 5,     # generations on each group of tracks before moving to the next
        'tracks_per_group': 2,   # tracks raced in parallel each generation
        'cycles': 2,             # passes through all the training groups
        'sigma': 0.05,           # std of the Gaussian noise added to mutated weights
        'mutation_rate': 1.0,    # fraction of weights that get noise in each mutant
        'max_time': 60.0,        # sim seconds per race
        'seed': 0,               # seed for gen-0 nets and mutation noise
        'tracks': None,          # training tracks, in order (None: every track that is not a test track)
        'test_tracks': ['YasMarina', 'Zandvoort'],  # held out; the best car races them after every cycle
        'val_tracks': ['Monza', 'Spielberg'],  # validation set; only `run.py --grid` races it, never trained on
        'score_metric': 'race_score', # what training ranks cars by: 'score' (general driving) or 'race_score' (pure speed, crash penalty)
        'shuffle_tracks': False, # shuffle the training track order once at the start
        'load': None,            # path to a weights.npz / best.npz to start from (None: random nets)
        'show_track': None,      # track to render the final best car on (None: first test track; 'none': skip)
        'workers': 0,            # worker processes (0: 75% of CPU threads)
    }
    return CONFIG


def get_grid():
    """@brief Values that `python run.py --grid` tries for each setting.

    Every combination of the listed values is trained and then scored on the
    validation tracks (config: val_tracks). The number of runs is the product of
    the list lengths, so keep the lists short. A setting with one value is held
    fixed; a setting left out takes its value from get_config().

    @return dict {setting name: list of values}. Names are keys of get_config():
            n, gens_per_group, tracks_per_group, cycles, sigma, mutation_rate, max_time.
    """
    return {
        'n': [200, 500],
        'gens_per_group': [3, 5],
        'tracks_per_group': [2],
        'cycles': [2],
        'sigma': [0.02, 0.05, 0.1],
        'mutation_rate': [0.25, 1.0],
        'max_time': [60.0],
    }
