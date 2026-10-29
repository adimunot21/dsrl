#!/usr/bin/env bash
# Stops this RunPod pod by itself so a forgotten pod cannot drain the budget (user authorised this, 2026-10-29).
# Usage: setsid nohup bash pod_watchdog.sh <hard_limit_minutes> > /root/watchdog.log 2>&1 &
# Stops when ANY of: (1) hard limit reached, (2) /root/STOP_NOW exists, (3) no work process for IDLE_MINUTES.
# "Work" = python/pip/conda/git/curl/tar/wget processes. Results must be synced off the pod BEFORE (2)/(3) fire:
# a stopped pod's container disk may not survive. At limit-15min a /root/WARN_DEADLINE file is written.
set -u
HARD_LIMIT_MIN="${1:?hard limit in minutes}"
IDLE_MINUTES=20          # idle grace: long enough for a sync after the last job, short enough to cap waste
CHECK_SECONDS=60
started=$(date +%s); idle_since=""
log() { echo "$(date -u +%FT%TZ) $*"; }
stop_pod() {
  log "STOPPING POD ($1)"
  for attempt in 1 2 3 4 5; do
    if runpodctl stop pod "$RUNPOD_POD_ID"; then log "stop requested ok"; sleep 120; fi
    log "stop attempt $attempt did not end the pod yet"; sleep 60
  done
  log "GIVING UP: stop failed -- user must stop the pod manually"; touch /root/WATCHDOG_FAILED
}
log "watchdog start: pod=$RUNPOD_POD_ID hard_limit=${HARD_LIMIT_MIN}min idle=${IDLE_MINUTES}min"
while true; do
  now=$(date +%s); elapsed_min=$(( (now - started) / 60 ))
  [ -f /root/STOP_NOW ] && stop_pod "STOP_NOW file"
  [ "$elapsed_min" -ge "$HARD_LIMIT_MIN" ] && stop_pod "hard limit ${HARD_LIMIT_MIN}min"
  [ "$elapsed_min" -ge $((HARD_LIMIT_MIN - 15)) ] && touch /root/WARN_DEADLINE
  if pgrep -x python >/dev/null || pgrep -x pip >/dev/null || pgrep -x conda >/dev/null \
     || pgrep -x git >/dev/null || pgrep -x curl >/dev/null || pgrep -x tar >/dev/null || pgrep -x wget >/dev/null || pgrep -f "bin/python|bin/pip|bin/conda" >/dev/null; then
    idle_since=""
  else
    [ -z "$idle_since" ] && idle_since=$now
    idle_min=$(( (now - idle_since) / 60 ))
    [ "$idle_min" -ge "$IDLE_MINUTES" ] && stop_pod "idle ${idle_min}min"
  fi
  sleep "$CHECK_SECONDS"
done
