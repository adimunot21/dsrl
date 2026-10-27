# Upstream DSRL build log (2026-10-27, laptop, no cloud)

Result: env WORKS for Robomimic Can and D4RL Hopper (env construction + random-action stepping through the repo's own
wrappers, SubprocVecEnv). `import train_dsrl` succeeds. No training run, no checkpoints downloaded. ~70 min elapsed.
Env: conda `dsrl_upstream`, Python 3.9.23. Exact versions: envs/dsrl_upstream.lock.txt. Recipe: envs/dsrl_upstream.md.
Shell setup used for every command: `LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:...`, `MUJOCO_GL=egl`,
`WANDB_MODE=disabled`, `D4RL_SUPPRESS_IMPORT_ERROR=1`, and `CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc` (system gcc is 13.3).

## 1. Submodules
`git -c url."https://github.com/".insteadOf=git@github.com: submodule update --init --recursive third_party/dsrl`
(.gitmodules untouched). Result: dppo 86ce51834055c02f9013e60dd4c4275606d82df7, stable-baselines3
10e5d311b5c42fe571ff3eeb72f622ca616a149d: both match the gitlink pins. Superproject third_party/dsrl at 8d21b9c.
Pip -e installs create egg-info that is git-ignored (`git status` clean in all three repos).

True pins (read, not assumed):
- dppo/pyproject.toml (no setup.py, no requirements.txt): av==12.3.0, einops==0.8.0, gdown==5.2.0, gym==0.22.0,
  hydra-core==1.3.2, imageio==2.35.1, matplotlib==3.7.5, omegaconf==2.3.0, pretty_errors==1.2.25, torch==2.4.0,
  tqdm==4.66.5, wandb==0.17.7. Extras gym: cython<3, d4rl (UNPINNED), patchelf. robomimic: + robomimic==0.3.0,
  robosuite @ git tag v1.4.1 (resolved c848ca8).
- stable-baselines3/setup.py: gymnasium>=0.29.1,<1.3.0, numpy>=1.20,<3, torch>=2.3, cloudpickle, pandas, matplotlib.
- Security read: dppo pyproject, SB3 setup.py, d4rl 1.1 sdist setup.py, robomimic 0.3.0 setup.py, robosuite v1.4.1 setup.py all
  plain setuptools, no os.system/subprocess/network in setup. Nothing suspicious found in them. Not audited: runtime code.
- Spec corrections: spec guessed mjrl commit/dm_control/egl_probe as needed (not needed for Can/Hopper); spec proposed
  numpy 1.26.4 (right) and wandb 0.17.7 (right, matches dppo); d4rl `PyPI d4rl 1.1` pulls dm_control, pybullet (skipped).

## 2. Build, in order
1. `curl -L .../deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz` -> ~/.mujoco/mujoco210 (4.4MB). OK.
2. `conda create -n dsrl_upstream -c conda-forge python=3.9 pip patchelf gcc_linux-64=12 gxx_linux-64=12 mesalib glew glfw libgl-devel libegl-devel`. OK.
   Note: conda gcc is named x86_64-conda-linux-gnu-gcc, plain `gcc` stays system gcc 13 -> CC must be exported.
3. `pip install "pip<24.1" "setuptools<66" "wheel<0.40" "Cython<3"` -> pip 24.0, setuptools 65.7.0, wheel 0.38.4, Cython 0.29.37. OK.
4. HAZARD (hang, not error): one combined `pip install torch==2.4.0 numpy gym==0.22.0 av ... wandb ...` ran >10 min at ~60% CPU
   with no progress (resolver/gym metadata on the combined set). Fix: install in separate steps (torch+numpy, gym, rest). Each was fast
   afterwards; torch 2.4.0 pulls cu121 wheels (~3GB).
5. mujoco-py 2.1.2.14 with CC=conda gcc 12: `import mujoco_py` compiled cymj.pyx once, no errors; second import prints 2.1.2.14. The
   GCC13/Cython3 hazard from the spec did not trigger (used gcc12 + Cython 0.29.37). Did not test whether system gcc 13 alone would fail.
6. `pip install --no-deps d4rl==1.1` + h5py termcolor click. `import gym, d4rl, d4rl.gym_mujoco` prints "No module named mjrl/flow/dm_control/
   carla/pybullet_envs" warnings (harmless; D4RL_SUPPRESS_IMPORT_ERROR=1 hides them). `gym.make('hopper-medium-v2')` works, obs (11,).
7. robosuite v1.4.1 (git clone --depth 1 --branch v1.4.1, reviewed) and robomimic 0.3.0 sdist installed `--no-deps`, with mujoco==2.3.7,
   numba scipy Pillow opencv-python-headless pynput psutil tensorboard tensorboardX imageio-ffmpeg torchvision==0.19.0. OK.
   mujoco-py (2.1 binary) and PyPI mujoco 2.3.7 coexist without clash.
8. gymnasium==1.1.1 cloudpickle pandas, then `pip install --no-deps -e dppo -e stable-baselines3`. OK (SB3 reports 2.6.0a1).
9. HAZARD: gymnasium/pandas step upgraded numpy 1.26.4 -> 2.0.2. Fix: `pip install numpy==1.26.4`, and do it last.
10. HAZARD: `import wandb` -> `ImportError: cannot import name 'Deprecated' from wandb.proto.wandb_telemetry_pb2`. Cause: protobuf 6.33.6
    pulled in by tensorboard 2.21 vs wandb 0.17.7 (needs protobuf<6). Fix: `protobuf<5` (4.25.9) + `tensorboard<2.18` (2.17.1).
11. HAZARD: opencv-python-headless 5.0.0.93 requires numpy>=2. Fix: `opencv-python-headless<4.11` (4.10.0.84).
12. `python -c "import train_dsrl"` -> import ok (only the d4rl missing-module notes).
Remaining `pip check` complaints, intentionally ignored: d4rl wants dm-control/pybullet, robomimic wants egl-probe, robosuite wants
opencv-python (headless satisfies the import).
robosuite prints "No private macro file found" warnings on every env construction (harmless).

## 3. Smoke test (envs/dsrl_upstream_bench.py; DUMMY normalization.npz, identity range -1..1, because the real one ships with the
Drive checkpoints that were not downloaded; so obs VALUES are not the real normalized ones)
Built through the repo's own make_robomimic_env + ObservationWrapperRobomimic/ObservationWrapperGym + ActionChunkWrapper
(act_steps=4) + make_vec_env(SubprocVecEnv), as in train_dsrl.py.
- Can (robosuite PickPlaceCan, 7-dim base action, chunk 4): obs (n,23) float64 at runtime although declared float32 box, values in about
  [-1.0, 1.04] with dummy norm; action space (28,) = 4x7 in [-1,1]; per-chunk reward with reward_offset=1 and random actions:
  exactly -4.0 always (4 substeps x -1, i.e. sparse task reward 0 never hit by random actions). Env construct ~9 s.
- Hopper (hopper-medium-v2, act chunk 4): obs (n,11) float64; action (12,) = 4x3 in [-1,1]; reward per chunk range about
  [-3.9, 14.0] (4 steps of dense reward summed); random policy terminates every ~6 chunks.
- Stdout noise: SB3 fork prints "env class: ..., MRO ..., Is instance" per env creation (debug print at
  stable-baselines3/stable_baselines3/common/vec_env/patch_gym.py:27). Harmless but spammy.

## 4. Raw throughput (NO training, random actions, SubprocVecEnv, includes resets and IPC)
CAVEAT: the machine was NOT idle: load average 8-17 on 12 threads (other agents' pytest + a training job). Treat as pessimistic,
perhaps 1.5-2x low. "chunk-step" = one env.step of the wrapper = 4 simulator control steps.
| env | n_envs | chunk-steps/s | env-steps/s (x4) |
|---|---|---|---|
| Hopper | 1 | 111-138 | 446-551 |
| Hopper | 4 | 156-268 | 625-1072 |
| Can | 1 | 2-3 | 8-13 |
| Can | 4 | 7-9 | 27-36 |
Profile of Can (cProfile, 6 chunk steps): ~93 ms per env step = 25 sim substeps each ~3.5 ms, dominated by robosuite's Python OSC
controller (osc.py run_controller + update, numpy) and mj_forward calls, not by rendering or IPC. Hopper at ~2 ms/env-step is
also far above raw mujoco-py cost, likely Python wrapper overhead plus contention. Re-measure on an idle machine before planning.
Implication (rough): 100k Can chunk-steps would take ~3-4 h at the n=4 rate measured under load, before any gradient cost;
DSRL's utd=20 with 3x2048 MLPs adds GPU work on a 4GB GTX 1650.

## 5. Not done / open
- Google Drive checkpoints and normalization.npz not downloaded (separate step); `python train_dsrl.py` not run.
- GPU path (torch.cuda on GTX 1650 with torch 2.4.0+cu121) not exercised here.
- wandb offline/disabled only checked via import.
