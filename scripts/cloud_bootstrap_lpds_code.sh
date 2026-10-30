#!/usr/bin/env bash
# Build the patched LP-DS code copy on a pod that already ran cloud_bootstrap_upstream.sh. LP-DS runs in the dsrl_upstream env
# (it already holds gym/d4rl/mujoco-py/robomimic/robosuite); lpds' own stable_baselines3 wins because the script dir is first on sys.path.
set -eo pipefail
cd "$HOME/dsrl"
git -c url."https://github.com/".insteadOf=git@github.com: submodule update --init third_party/lpds
rm -rf "$HOME/lpds_patched"; cp -r third_party/lpds "$HOME/lpds_patched"
(cd "$HOME/lpds_patched" && patch -p1 < "$HOME/dsrl/patches/0004_lpds_eval_log.patch")
sha256sum "$HOME/lpds_patched/artifacts/backbones/gym/hopper-medium-v2/state_3000.pt" "$HOME/lpds_patched/artifacts/normalization/gym/hopper-medium-v2/normalization.npz"
cd "$HOME/lpds_patched"
"$HOME/miniforge3/envs/dsrl_upstream/bin/python" -I -c "
import sys; sys.path.insert(0,'.')
import env_utils, stable_baselines3
from stable_baselines3 import LPDS
print('LPDS import ok from', stable_baselines3.__file__)"
echo "lpds code ok"
