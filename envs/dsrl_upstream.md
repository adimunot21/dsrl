# Upstream DSRL environment (BUILT 2026-10-27, smoke-tested; no training run)

Full command-by-command record: docs/upstream_build_log.md. Exact versions: dsrl_upstream.lock.txt.
Throughput script: envs/dsrl_upstream_bench.py (uses a DUMMY normalization.npz; real one is in the Drive checkpoints, not downloaded).

Submodules (checked out over HTTPS, match gitlink pins): dppo 86ce518, stable-baselines3 10e5d31.
True pins come from dppo/pyproject.toml (torch 2.4.0, gym 0.22.0, hydra-core 1.3.2, omegaconf 2.3.0, wandb 0.17.7,
av 12.3.0, einops 0.8.0, ...; extras: cython<3, d4rl UNPINNED, robomimic==0.3.0, robosuite git tag v1.4.1) and
stable-baselines3/setup.py (gymnasium>=0.29.1,<1.3, torch>=2.3, numpy>=1.20,<3).

## Verified build recipe
```
git -c url."https://github.com/".insteadOf=git@github.com: submodule update --init --recursive third_party/dsrl
mkdir -p ~/.mujoco; curl -L https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz | tar xz -C ~/.mujoco
conda create -n dsrl_upstream -c conda-forge python=3.9 pip patchelf gcc_linux-64=12 gxx_linux-64=12 mesalib glew glfw libgl-devel libegl-devel
conda activate dsrl_upstream
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:$LD_LIBRARY_PATH MUJOCO_GL=egl WANDB_MODE=disabled
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc D4RL_SUPPRESS_IMPORT_ERROR=1
pip install "pip<24.1" "setuptools<66" "wheel<0.40" "Cython<3"
pip install torch==2.4.0 numpy==1.26.4 ; pip install gym==0.22.0
pip install av==12.3.0 einops==0.8.0 gdown==5.2.0 hydra-core==1.3.2 imageio==2.35.1 matplotlib==3.7.5 omegaconf==2.3.0 pretty_errors==1.2.25 tqdm==4.66.5 wandb==0.17.7
pip install mujoco-py==2.1.2.14 cffi "glfw<2.8" lockfile ; python -c "import mujoco_py"   # compiles, ~1-2 min
pip install --no-deps d4rl==1.1 ; pip install h5py termcolor click
pip install mujoco==2.3.7 numba scipy Pillow opencv-python-headless pynput psutil tensorboard tensorboardX imageio-ffmpeg torchvision==0.19.0 gymnasium==1.1.1 cloudpickle pandas
pip install --no-deps <local clone of robosuite @ v1.4.1> robomimic==0.3.0
pip install --no-deps -e third_party/dsrl/dppo -e third_party/dsrl/stable-baselines3
pip install "numpy==1.26.4" "protobuf<5" "opencv-python-headless<4.11" "tensorboard<2.18"   # repair step, do LAST
```
Why --no-deps: dppo extras pull d4rl (-> dm_control, pybullet, mjrl) and robomimic (-> egl_probe needing cmake) that are not needed for Can/Hopper.

## Remaining caveats
- Checkpoints + normalization.npz (Google Drive) not yet downloaded.
- Can env is slow on CPU (python OSC controller, 25 substeps per env step); see log for numbers.
- SB3 fork has a debug print ("env class: ...") in common/vec_env/patch_gym.py:27, spams stdout per env.
- pip check still reports unused-package gaps (dm-control, pybullet, egl-probe, opencv-python): intentional.
- Hazards 3/4/5 of the old spec were confirmed in part; see log for what actually bit.
