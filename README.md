# FSAE Autonomous Car Sim

A Python simulation environment for prototyping autonomous-vehicle control algorithms for an FSAE (Formula SAE) car, built on a fork of the [F1TENTH Gym](https://github.com/Anna-Kopec/f1tenth_gym).

## Why F1TENTH?

F1TENTH is a lightweight simulator that handles the vehicle physics, so you can focus on implementing the control algorithms themselves.

- **Python only.** No heavy rendering engines (Unreal, Unity) are required, unlike many other simulators.
- **Easy to set up.** A Python 3.10 environment and a single `pip install` are enough to get running.
- **Well suited to prototyping.** Fast iteration on planning and control before moving to higher-fidelity tools.

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10 | **Do not use 3.11 or newer** (including the 3.12/3.13 that many systems ship with): the simulator needs `gym<0.24` and `numpy<2`, which do not support them. Download 3.10 from <https://www.python.org/downloads/> (on Windows, tick "Add python.exe to PATH" or keep the `py` launcher option on). |
| `git` | To clone the repository. |

Nothing else needs installing: the virtual environment is made with Python's built-in `venv` module.

## Quick Start

Commands below are the same on every system except where a table shows one column per system. Use PowerShell or cmd on Windows, and Terminal on macOS and Linux.

### 1. Clone the repository

```bash
git clone [https://github.com/uconnformulasae/autonomous-sim-prototype](https://github.com/uconnformulasae/autonomous-sim-prototype)
cd autonomous-sim-prototype
```

### 2. Create the virtual environment

It must be made with Python 3.10.

| Windows | macOS / Linux |
|---|---|
| `py -3.10 -m venv gym_env` | `python3.10 -m venv gym_env` |

If `py` or `python3.10` is not found, Python 3.10 is not installed (or not on your `PATH`); see Prerequisites. If a `gym_env` folder already exists (for example one made on another operating system), choose another name such as `gym_env_win`, or delete the old folder, and use that name in the next step.

### 3. Activate the environment and install the simulator

| Windows (PowerShell) | Windows (cmd) | macOS / Linux |
|---|---|---|
| `gym_env\Scripts\Activate.ps1` | `gym_env\Scripts\activate.bat` | `source gym_env/bin/activate` |

Check that it is Python 3.10 (`python --version`), then, from the repository root, on every system:

```bash
python -m pip install --upgrade pip
pip install -e .
```

Use `deactivate` to exit the environment.

### 4. Run the simulation

Run from the repository root with the environment active.

| macOS | Windows |
|---|---|
| `python genetic_sim/testing_env.py` | `python genetic_sim\testing_env.py` |

This races 10 random cars on one track in a render window. Add `--headless` for no window. Training is covered in the next section.

### 5. Train a controller

The controller is a small neural net whose weights are evolved: each generation races a population of cars, the best car is kept, and mutated copies of it become the next generation. All commands below are run from the repository root with the environment active. Paths are shown for Windows; on macOS/Linux use `/` instead of `\`.

Tracks are split three ways, and the test and validation tracks are never trained on:

| Set | Used for |
|---|---|
| Training tracks | Evolving the weights (every track not listed as test or validation, unless `tracks` is set) |
| Test tracks (`test_tracks`) | The best car races them after every cycle; scores are logged only |
| Validation tracks (`val_tracks`) | Only `--grid` races them, to rank settings |

**Step 1: choose settings.** Edit `genetic_sim/config.py`. Each key is commented there (population size `n`, `sigma`, `mutation_rate`, `gens_per_group`, `cycles`, `max_time`, the track lists, and so on). `get_grid()` in the same file lists the values the grid search tries.

**Step 2 (optional): grid search for good settings.** Trains once per combination in `get_grid()`, scores each result on the validation tracks, and saves the best one:

```
python genetic_sim\run.py --grid
```

Output goes to `genetic_sim/runs/gridsearch_<timestamp>/`: `grid_search.log`, `best_config.json` (the winning settings) and `best.npz` (its weights). Copy the winning values into `get_config()` for step 3.

**Step 3: train.**

```
python genetic_sim\run.py
```

Output goes to `genetic_sim/runs/genetic_<timestamp>/`, rewritten every generation, so stopping with Ctrl+C keeps everything up to the last finished generation:

| File | Contents |
|---|---|
| `config.json` | The settings the run used |
| `history.csv`, `generations.csv` | Best and mean score per generation; every car's score |
| `test.csv` | Test-track results after each cycle |
| `best.npz` | Best car of the latest generation |
| `best_cycleN.npz` | Best car at the end of cycle N |
| `population.npz` | The whole latest population |

When training ends, the final best car drives `show_track` in a render window. Set `show_track` to `'none'` in the config to skip that.

**Step 4: continue from saved weights.** `--resume` starts a new run from existing weights, with the exact settings of the run they came from (read from the `config.json` or `best_config.json` beside the file, ignoring `config.py`):

```
python genetic_sim\run.py --resume genetic_sim\runs\genetic_<timestamp>\best.npz
```

A single-car file (`best.npz`, `best_cycleN.npz`) becomes generation 0 as that car plus mutants of it. A file holding exactly `n` cars (`population.npz`) is used as generation 0 unchanged. A new `genetic_<timestamp>` folder is created and the old one is untouched. To resume with different settings instead, set `load` in `config.py` to the weights path and run `python genetic_sim\run.py`. `--resume` cannot be combined with `--grid`.

**Step 5: replay weights with visuals and scores.** `replay.py` drives a saved car on each track in the render window, then prints a score table and saves it as a csv next to the weights:

```
python genetic_sim\replay.py genetic_sim\runs\genetic_<timestamp>\best.npz
```

| Option | Effect |
|---|---|
| (none) | Races the test tracks from the run's `config.json` (or `config.py` if there is none; grid-search folders have `best_config.json`, so they use `config.py`) |
| `--val` | Also race the validation tracks |
| `--tracks Monza Spa` | Race exactly these tracks |
| `--all` | Race every track |
| `--car N` | Pick row N of a multi-car file (default: the best car) |
| `--max-time 90` | Sim seconds per track (default: the run's `max_time`, else 60) |
| `--hold 2` | Seconds each finished race stays on screen |
| `--headless` | No window, scores only (much faster) |
| `--no-save` | Do not write the results csv |

Closing the window ends the visuals; the remaining tracks are still scored without it. To watch a saved car on one track in the original multi-car viewer instead, use `python genetic_sim\testing_env.py --load <weights.npz> --map Monza`.

A typical workflow is: grid search, copy the best settings into `config.py`, train, replay `best.npz` with `--val` to check it generalises, then `--resume` to train further.

## Documentation

### Read Write-ups
Formal written documentation for this project can be found in docs/writeups. 
- [Autonomous Car Guidelines](<docs/writeups/Autonomous Car Guidelines.pdf>): UConn FSAE's autonomous vehicle initiative overview
- [CSE 4820 Project Proposal](docs/writeups/CSE_4820_Course_Project_Proposal.pdf): Our team's project proposal (this is also doubling as a class project)
- [CSE 4820 Project Progress Report](docs/writeups/CSE_4820_Course_Project_Progress_report.pdf): Our team's project progress report

### Viewing Doxygen Code Documentation

Open `docs/doxygen/html/index.html` in a browser.

| macOS | Windows (PowerShell) |
|---|---|
| `open docs/doxygen/html/index.html` | `start docs\doxygen\html\index.html` |

You can also double-click `index.html` in Finder or File Explorer. If search or navigation breaks when opened as a local file, serve the folder and visit `http://localhost:8000`:

| macOS | Windows (PowerShell) |
|---|---|
| `python3 -m http.server 8000 --directory docs/doxygen/html` | `python -m http.server 8000 --directory docs\doxygen\html` |

## Troubleshooting

| Issue | Fix |
|---|---|
| `command not found` / "not recognized" (e.g., `git`, `py`, `python3.10`) | Install the missing prerequisite and make sure it is on your `PATH`. `py` only exists on Windows when Python came from python.org. |
| PowerShell: "running scripts is disabled" when activating | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then activate again (or use `activate.bat` in cmd). |
| `Numba needs NumPy 2.3 or less`, or pip fails building `numpy`/`pyglet`/`gym` | The environment is on a Python that is too new. Check `python --version`; recreate it with Python 3.10. |
| `No module named 'gym'` / `f110_gym` | The environment is not active, or `pip install -e .` was not run from the repository root. |
| Folder `gym_env` already exists and has `bin/` instead of `Scripts/` | It was made on Linux/macOS (or WSL). It cannot run on Windows; use another folder name. |
| macOS (Big Sur+) with rendering on: `ImportError: Can't find framework /System/Library/Frameworks/OpenGL.framework` | Run `pip install pyglet==1.5.20`. |
| `f110-gym requires pyglet<1.5, but you have pyglet 1.5.20` | Safe to ignore; the environment should still work. |