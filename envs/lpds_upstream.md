# LP-DS upstream environment (SPEC ONLY, not built)

Source: third_party/lpds/environment.yml, requirements.txt, setup.py (read only, untrusted).
Install per its README: `conda env create -f environment.yml` (pip section runs `-r requirements.txt` and `-e .`).

## Hazards
1. Package collision: lpds ships `stable_baselines3` (LP-DS fork, `stable_baselines3.LPDS`) plus `lpds_models`; DSRL's SB3 fork has the same top-level name. Separate env mandatory; never put both on PYTHONPATH.
2. Same mujoco-py 2.1.2.14 hazards as dsrl_upstream.md (MuJoCo 2.1 binaries, gcc<=12, Cython==0.29.37 pinned, LD_LIBRARY_PATH, GL libs).
3. Both mujoco==3.3.5 and mujoco-py 2.1 are pinned: different libraries, coexist but double-check.
4. gym==0.22.0 needs old pip/setuptools/wheel to build; the environment.yml does not pin them, so `conda env create` may fail at the pip stage. Pre-pin pip<24.1, setuptools<66, wheel<0.40.
5. d4rl==1.1 (PyPI) drags mujoco-py/dm_control; mjrl is a git install (network, pinned commit 3871d93).
6. setup.py install_requires duplicates the pins (including git URL), so `-e .` re-resolves them; use `pip install -e . --no-deps` after requirements.
7. Pins are heavy: torch 2.4.0 (default PyPI wheel bundles CUDA 12.1; fine on GTX 1650 / driver 595), pinocchio from conda-forge may conflict with python 3.9 pins.
8. wandb==0.17.7 pinned; check whether lpds imports it unconditionally (set WANDB_MODE=disabled).
9. defaults channel in environment.yml may trigger Anaconda ToS / mixed-channel solver problems; prefer conda-forge only with strict priority.
10. Weights/artifacts in third_party/lpds/artifacts are untrusted pickles/checkpoints: torch.load with weights_only=True.

## Build order
Same as dsrl_upstream.md steps 1-5 (conda base libs, pip pins, MuJoCo 2.1, mujoco-py compile test, requirements.txt), then `pip install -e third_party/lpds --no-deps`, then smoke import `from stable_baselines3 import LPDS`.
