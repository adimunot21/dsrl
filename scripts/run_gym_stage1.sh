#!/usr/bin/env bash
# Stage 1 of the Gym phase (configs/gym_budget.json): arms A and C, Hopper, reporting seeds 0-9, 10 runs at a time.
# Interleaved so every wave holds both arms. Logs: /root/runs/stage1/<arm>_s<seed>/
set -eo pipefail
cd "$HOME/dsrl"
jobs_file=$(mktemp)
for seed in 0 1 2 3 4 5 6 7 8 9; do echo "orig $seed A"; echo "patched $seed C"; done > "$jobs_file"
cat "$jobs_file" | xargs -P 10 -L 1 bash -c '
  code=$0; seed=$1; arm=$2
  root=/root/runs/stage1/${arm}_s${seed}
  mkdir -p "$root"
  start=$(date +%s)
  bash scripts/run_gym_arm.sh "$code" hopper "$seed" "$root" algorithm=dsrl_sac +total_raw_env_steps=200000 > "$root/stdout.txt" 2>&1 \
    && echo "$(date -u +%FT%TZ) DONE arm=$arm seed=$seed wall_s=$(( $(date +%s) - start ))" >> /root/runs/stage1/status.log \
    || echo "$(date -u +%FT%TZ) FAIL arm=$arm seed=$seed" >> /root/runs/stage1/status.log'
echo "$(date -u +%FT%TZ) STAGE1 DONE" >> /root/runs/stage1/status.log
