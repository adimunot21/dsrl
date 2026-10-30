#!/usr/bin/env bash
# Stage 3 (prereg 002c): arm B = LP-DS as released, Hopper, seeds 10-19, 5 concurrent. Logs: /root/runs/stage3/B_s<seed>/
set -eo pipefail
cd "$HOME/dsrl"
mkdir -p /root/runs/stage3
printf '%s\n' 10 11 12 13 14 15 16 17 18 19 | xargs -P 5 -I{} bash -c '
  seed={}; root=/root/runs/stage3/B_s${seed}; mkdir -p "$root"; start=$(date +%s)
  LPDS_PY=$HOME/miniforge3/envs/dsrl_upstream/bin/python bash scripts/run_gym_lpds.sh "$seed" "$root" total_timesteps=50000 eval_interval=1000 > "$root/stdout.txt" 2>&1 \
    && echo "$(date -u +%FT%TZ) DONE arm=B seed=$seed wall_s=$(( $(date +%s) - start ))" >> /root/runs/stage3/status.log \
    || echo "$(date -u +%FT%TZ) FAIL arm=B seed=$seed" >> /root/runs/stage3/status.log'
echo "$(date -u +%FT%TZ) STAGE3 DONE" >> /root/runs/stage3/status.log
