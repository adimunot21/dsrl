# Upstream DSRL environment (SPEC ONLY, not built)

Source: third_party/dsrl/README.md. Documented install: `conda create -n dsrl python=3.9`, then
`pip install -e dppo; pip install -e dppo[robomimic]; pip install -e dppo[gym]; pip install -e stable-baselines3`.

Submodules (not checked out; `git submodule status`):
- dppo           86ce51834055c02f9013e60dd4c4275606d82df7
- stable-baselines3 10e5d311b5c42fe571ff3eeb72f622ca616a149d
Both directories are empty, so exact dependency pins are UNKNOWN until fetched. dsrl_upstream.yml is a
proposal modeled on lpds pins. Checkpoints must be downloaded by hand from Google Drive into dppo/log.

## Hazards
1. Submodules empty: `git submodule update --init` needed (network; untrusted code - inspect setup.py before pip install -e, which executes it).
2. Python 3.9 is EOL; old wheels (mujoco-py, gym 0.2x, robosuite 1.4) may have no wheels for newer deps.
3. mujoco-py 2.1.2.14 needs MuJoCo 2.1.0 binaries in ~/.mujoco/mujoco210 (manual download from github.com/deepmind/mujoco/releases), LD_LIBRARY_PATH incl. mujoco210/bin and the NVIDIA lib dir, gcc, patchelf, GL libs (libgl1-mesa-dev, libosmesa6-dev, libglew-dev). It compiles on first import (slow, fails with GCC 13 / Cython 3: use Cython<3, gcc<=12).
4. gym 0.21/0.22 metadata is invalid for modern pip/setuptools/wheel: pin pip<24.1, setuptools<66, wheel<0.40.
5. d4rl: PyPI d4rl 1.1 pulls mujoco-py, mjrl (git URL, lpds pins commit 3871d93) and dm_control; install with --no-deps and add deps manually. d4rl imports may try to download datasets and print warnings about missing Flow/CARLA/mujoco.
6. Both mujoco (3.x, used by dm-control/robosuite 1.4... robosuite 1.4 needs mujoco>=2.3.2) and mujoco-py (2.1) coexist: confirm no clash.
7. gym vs gymnasium: SB3 fork may expect one API, dppo gym envs the other; both may be installed (lpds pins gym 0.22.0 + gymnasium 1.1.1).
8. robomimic 0.3.0 / robosuite 1.4.1 need specific commits/versions of each other; robosuite needs mujoco GL (set MUJOCO_GL=egl or osmesa on headless).
9. wandb imported unconditionally at top of train_dsrl.py (line 7) even if use_wandb=false: must be installed; set WANDB_MODE=disabled. Also cfg may hold WANDB entity/ API key requirements.
10. Hydra config paths relative (`--config-path=cfg/...`), run from repo root; dppo reads env vars like DPPO_DATA_DIR / DPPO_LOG_DIR (verify once fetched).
11. torch version must support GTX 1650 (sm_75) - fine for cu118-cu124 wheels; the system CUDA 13.2 driver is backwards compatible.
12. Name collision: SB3 fork installs package `stable_baselines3`; cannot share an env with lpds (see lpds_upstream.md).

## Proposed build order
1. conda create from dsrl_upstream.yml conda part (python 3.9, gcc 12, GL libs, patchelf) - no pip section yet.
2. pip pin: `pip install "pip<24.1" "setuptools<66" "wheel<0.40" "Cython<3"`.
3. Install MuJoCo 2.1 binaries to ~/.mujoco/mujoco210; export LD_LIBRARY_PATH; MUJOCO_GL=egl.
4. torch, gym==0.22.0, mujoco-py; `python -c "import mujoco_py"` to force compile.
5. d4rl --no-deps, mjrl, dm-control, then hydra/omegaconf/wandb.
6. `git submodule update --init`; review setup.py files; pip install -e dppo[robomimic,gym]; pip install -e stable-baselines3.
7. Smoke test: import chain, one gym env step, then `WANDB_MODE=disabled python train_dsrl.py ...` for a few steps.
