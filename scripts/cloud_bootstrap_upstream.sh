#!/usr/bin/env bash
# Build the Python 3.9 upstream-DSRL env on a fresh Linux pod (recipe: envs/dsrl_upstream.md, verified on the laptop 2026-10-27).
# Run AFTER scripts/cloud_bootstrap.sh has cloned the repo to ~/dsrl:  bash scripts/cloud_bootstrap_upstream.sh
# Pip installs go in SEPARATE steps on purpose (one combined resolve hung >10 min in the laptop build).
set -eo pipefail
# mujoco-py compiles against GL/osmesa.h (system headers on the laptop; absent on a fresh pod)
(apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libosmesa6-dev libgl1-mesa-dev libglew-dev libglfw3 patchelf) || echo "apt step failed; mujoco-py compile may fail"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
cd "$HOME/dsrl"
git -c url."https://github.com/".insteadOf=git@github.com: submodule update --init --recursive third_party/dsrl
mkdir -p ~/.mujoco
[ -d ~/.mujoco/mujoco210 ] || curl -L https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz | tar xz --no-same-owner -C ~/.mujoco
conda env list | grep -q "^dsrl_upstream " || conda create -y -n dsrl_upstream -c conda-forge python=3.9 pip patchelf gcc_linux-64=12 gxx_linux-64=12 mesalib glew glfw libgl-devel libegl-devel
conda activate dsrl_upstream
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH:-} MUJOCO_GL=egl WANDB_MODE=disabled
export CPATH=/usr/include:/usr/include/x86_64-linux-gnu LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc D4RL_SUPPRESS_IMPORT_ERROR=1
pip install "pip<24.1" "setuptools<66" "wheel<0.40" "Cython<3"
pip install torch==2.4.0 numpy==1.26.4
pip install gym==0.22.0
pip install av==12.3.0 einops==0.8.0 gdown==5.2.0 hydra-core==1.3.2 imageio==2.35.1 matplotlib==3.7.5 omegaconf==2.3.0 pretty_errors==1.2.25 tqdm==4.66.5 wandb==0.17.7
pip install mujoco-py==2.1.2.14 cffi "glfw<2.8" lockfile
python -c "import mujoco_py; print('mujoco_py', mujoco_py.__version__)"
pip install --no-deps d4rl==1.1
pip install h5py termcolor click
pip install mujoco==2.3.7 numba scipy Pillow opencv-python-headless pynput psutil tensorboard tensorboardX imageio-ffmpeg torchvision==0.19.0 gymnasium==1.1.1 cloudpickle pandas
[ -d /root/robosuite ] || git clone --depth 1 --branch v1.4.1 https://github.com/ARISE-Initiative/robosuite /root/robosuite
pip install --no-deps /root/robosuite robomimic==0.3.0
pip install --no-deps -e third_party/dsrl/dppo -e third_party/dsrl/stable-baselines3
pip install "numpy==1.26.4" "protobuf<5" "opencv-python-headless<4.11" "tensorboard<2.18"   # repair step, LAST
# Code copies (patches/ applied to clean copies; the submodule itself stays untouched):
#   dsrl_base    = pinned + 0002 raw-step stop rule + 0003 local eval log   (arms A, F; neither changes learning)
#   dsrl_patched = dsrl_base + 0001 truncation fix                          (arms C, D, G)
for d in dsrl_base dsrl_patched; do rm -rf "/root/$d"; cp -r third_party/dsrl "/root/$d"; done
for d in dsrl_base dsrl_patched; do for pf in 0002_raw_step_budget 0003_eval_log; do (cd "/root/$d" && patch -p1 < "$HOME/dsrl/patches/$pf.patch"); done; done
(cd /root/dsrl_patched && patch -p1 < "$HOME/dsrl/patches/0001_truncation_fix.patch")
python - <<'PY'
import sys; sys.path.insert(0, "/root/dsrl/third_party/dsrl"); sys.path.append("/root/dsrl/third_party/dsrl/dppo")
import gym, d4rl, d4rl.gym_mujoco, torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "hopper obs", gym.make("hopper-medium-v2").reset().shape)
PY
echo "upstream bootstrap ok"
