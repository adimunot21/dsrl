#!/usr/bin/env bash
# Launch ONE LP-DS Gym run (arm B: LP-DS as released, default delta) from a patched copy of third_party/lpds.
# usage: run_gym_lpds.sh <seed> <log_root> [extra hydra overrides...]   (Hopper only in this stage)
# LP-DS ships its own copy of the Hopper checkpoint; its sha256 equals DSRL's (docs/checkpoint_manifest.json), so it is used as shipped.
set -eo pipefail
SEED="$1"; LOGROOT="$2"; shift 2
SRC="${LPDS_DIR:-$HOME/lpds_patched}"          # lpds + patch 0004 (logging only)
PY="${LPDS_PY:-$HOME/miniforge3/envs/lpds_upstream/bin/python}"
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH:-} MUJOCO_GL=egl WANDB_MODE=disabled D4RL_SUPPRESS_IMPORT_ERROR=1
export DSRL_EVAL_LOG="$LOGROOT/eval.jsonl"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} MKL_NUM_THREADS=${OMP_NUM_THREADS:-1}
mkdir -p "$LOGROOT"; cd "$SRC"
exec "$PY" train_lpds.py --config-path "$SRC/cfg/gym" --config-name lpds_hopper.yaml \
  use_wandb=False seed="$SEED" log_dir="$LOGROOT" "$@"
