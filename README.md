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
git clone https://github.com/uconnformulasae/autonomous-sim-prototype
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

This races 10 random cars on one track in a render window. Add `--headless` for no window. To train a controller, edit `genetic_sim/config.py`, then:

| What | Command (Windows paths shown; use `/` on macOS) |
|---|---|
| Train with the config | `python genetic_sim\run.py` |
| Grid search over `get_grid()`, ranked on the validation tracks | `python genetic_sim\run.py --grid` |

Results are written to `genetic_sim/runs/`.
## Documentation

### Read Write-ups
Formal written documentation for this project can be found in docs/writeups. 
* [Autonomous Car Guidelines](<docs/writeups/Autonomous Car Guidelines.pdf>): UConn FSAE's autonomous vehicle initiative overview
* [CSE 4820 Project Proposal](docs/writeups/CSE_4820_Course_Project_Proposal.pdf): Our team's project proposal (this is also doubling as a class project)
* [CSE 4820 Project Progress Report](docs/writeups/CSE_4820_Course_Project_Progress_report.pdf): Our team's project progress report

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
