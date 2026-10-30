#!/usr/bin/env bash
# Stage 2 retry (LOG 2026-10-30): the 15 runs that crashed at startup under 10-way concurrency on 18 vCPUs are re-run 5 at a time.
# Crashed runs wrote no eval.jsonl, so the retry set is not selected on outcomes. Same seeds, same arms, same overrides.
set -eo pipefail
cd "$HOME/dsrl"
mkdir -p /root/runs/stage2
for d in A_s10 A_s11 A_s12 A_s13 A_s14 A_s16 A_s17 A_s18 C_s10 C_s11 C_s12 C_s13 C_s14 C_s16 C_s17; do
  [ -d /root/runs/stage2/$d ] && mv /root/runs/stage2/$d /root/runs/stage2/${d}_crashed1
done
jobs_file=$(mktemp)
for seed in 10 11 12 13 14 16 17 18; do echo "orig $seed A"; done > "$jobs_file"
for seed in 10 11 12 13 14 16 17; do echo "patched $seed C"; done >> "$jobs_file"
cat "$jobs_file" | xargs -P 5 -L 1 bash -c '
  code=$0; seed=$1; arm=$2
  root=/root/runs/stage2/${arm}_s${seed}
  mkdir -p "$root"
  start=$(date +%s)
  bash scripts/run_gym_arm.sh "$code" hopper "$seed" "$root" algorithm=dsrl_sac +total_raw_env_steps=200000 eval_interval=1000 > "$root/stdout.txt" 2>&1 \
    && echo "$(date -u +%FT%TZ) DONE arm=$arm seed=$seed wall_s=$(( $(date +%s) - start ))" >> /root/runs/stage2/status.log \
    || echo "$(date -u +%FT%TZ) FAIL2 arm=$arm seed=$seed" >> /root/runs/stage2/status.log'
echo "$(date -u +%FT%TZ) STAGE2 RETRY DONE" >> /root/runs/stage2/status.log
