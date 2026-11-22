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
        'shuffle_tracks': False, # shuffle the training track order once at the start
        'load': None,            # path to a weights.npz / best.npz to start from (None: random nets)
        'show_track': None,      # track to render the final best car on (None: first test track; 'none': skip)
        'workers': 0,            # worker processes (0: 75% of CPU threads)
    }
    return CONFIG
