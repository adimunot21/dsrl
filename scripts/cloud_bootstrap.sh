#!/usr/bin/env bash
# Bootstrap a fresh Linux GPU pod for the toy-grid benchmark (prereg 001c). Run as: bash scripts/cloud_bootstrap.sh
# Untrusted-network rules: clone over HTTPS (no keys on the pod), no credentials stored.
set -euo pipefail
REPO_URL="https://github.com/adimunot21/dsrl.git"
if ! command -v conda >/dev/null; then
  wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/mf.sh
  bash /tmp/mf.sh -b -p "$HOME/miniforge3"
fi
source "$HOME/miniforge3/etc/profile.d/conda.sh" 2>/dev/null || true
[ -d dsrl ] || git clone "$REPO_URL" dsrl
cd dsrl
conda create -y -n dsrlstudy --file envs/dsrlstudy.explicit.txt
conda activate dsrlstudy
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
export PYTHONPATH=$PWD
python -m pytest -q tests -x
echo "bootstrap ok"
