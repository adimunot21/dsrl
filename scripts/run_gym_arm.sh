#!/usr/bin/env bash
# Launch ONE Gym DSRL run from a pinned (A, F, B...) or patched (C, D, G...) copy of the upstream code.
# usage: run_gym_arm.sh <code_dir: orig|patched> <task: hopper|walker|halfcheetah> <seed> <log_root> [extra hydra overrides...]
# Paths are passed on the command line so third_party/dsrl is never modified.
set -eo pipefail
CODE="$1"; TASK="$2"; SEED="$3"; LOGROOT="$4"; shift 4
REPO=${REPO:-$HOME/dsrl}
case "$CODE" in orig) SRC="${BASE_DIR:-$HOME/dsrl_base}";; patched) SRC="${PATCHED_DIR:-$HOME/dsrl_patched}";; *) echo "bad code dir"; exit 2;; esac
case "$TASK" in
  hopper)      ENV=hopper-medium-v2;      CFG=dsrl_hopper.yaml;      CKPT=$(ls $REPO/data/dppo_log/gym-pretrain/hopper-medium-v2_*/*/checkpoint/state_3000.pt);;
  walker)      ENV=walker2d-medium-v2;    CFG=dsrl_walker.yaml;      CKPT=$(ls $REPO/data/dppo_log/gym-pretrain/walker2d-medium-v2_*/*/checkpoint/state_3000.pt);;
  halfcheetah) ENV=halfcheetah-medium-v2; CFG=dsrl_halfcheetah.yaml; CKPT=$(ls $REPO/data/dppo_log/gym-pretrain/halfcheetah-medium-v2_*/*/checkpoint/state_3000.pt);;
  *) echo "bad task"; exit 2;;
esac
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH:-} MUJOCO_GL=egl WANDB_MODE=disabled D4RL_SUPPRESS_IMPORT_ERROR=1
export DSRL_EVAL_LOG="$LOGROOT/eval.jsonl"   # patch 0003: local eval log
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} MKL_NUM_THREADS=${OMP_NUM_THREADS:-1}
mkdir -p "$LOGROOT"; cd "$SRC"
exec "$HOME/miniforge3/envs/dsrl_upstream/bin/python" train_dsrl.py --config-path "$SRC/cfg/gym" --config-name "$CFG" \
  use_wandb=False seed="$SEED" log_dir="$LOGROOT" base_policy_path="$CKPT" \
  normalization_path="$REPO/data/dppo_log/gym/$ENV/normalization.npz" "$@"
