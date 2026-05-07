#!/bin/bash
# oom_guard.sh — cgroup-aware memory watchdog.
#
# Polls /sys/fs/cgroup/memory.current every 15s. Logs OK/WARN/KILL and
# auto-pkills deepspeed+train.py when the pod hits 85% of its memory.max.
# We kill at 85% (with SIGTERM, then SIGKILL after 5s) instead of letting
# the kernel OOM-kill at 100% — kernel OOMKill is SIGKILL-only, races with
# screen, and tends to nuke the entire training session ungracefully.
#
# Usage:
#   bash scripts/oom_guard.sh              # default thresholds (70% warn / 85% kill)
#   WARN_PCT=60 KILL_PCT=80 bash scripts/oom_guard.sh
#
# Run this in a separate screen window before launching Stage 10/20.
set -u

WARN_PCT="${WARN_PCT:-70}"
KILL_PCT="${KILL_PCT:-85}"
INTERVAL="${INTERVAL:-15}"

# Read cgroup memory limit (cgroup v2 path).
LIMIT_RAW=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo max)
if [[ "$LIMIT_RAW" == "max" || -z "$LIMIT_RAW" ]]; then
    # Fallback: assume 1800 GiB if cgroup is unbounded (best guess for this pod).
    LIMIT=$((1800 * 1024**3))
    echo "guard: cgroup limit unbounded, assuming 1800 GiB"
else
    LIMIT="$LIMIT_RAW"
fi

WARN=$((LIMIT * WARN_PCT / 100))
KILL=$((LIMIT * KILL_PCT / 100))

echo "guard: limit=$((LIMIT/1024**3))GiB  warn@${WARN_PCT}%=$((WARN/1024**3))GiB  kill@${KILL_PCT}%=$((KILL/1024**3))GiB  interval=${INTERVAL}s"

while true; do
    USED=$(cat /sys/fs/cgroup/memory.current 2>/dev/null || echo 0)
    PCT=$((USED * 100 / LIMIT))
    TS=$(date +%H:%M:%S)
    if [[ "$USED" -gt "$KILL" ]]; then
        echo "[$TS]  KILL  used=$((USED/1024**3))GiB  ${PCT}%  -> killing deepspeed/python"
        pkill -SIGTERM -f deepspeed
        pkill -SIGTERM -f 'python.*train'
        sleep 5
        pkill -SIGKILL -f deepspeed
        pkill -SIGKILL -f 'python.*train'
        exit 1
    elif [[ "$USED" -gt "$WARN" ]]; then
        echo "[$TS]  WARN  used=$((USED/1024**3))GiB  ${PCT}%"
    else
        echo "[$TS]  ok    used=$((USED/1024**3))GiB  ${PCT}%"
    fi
    sleep "$INTERVAL"
done
