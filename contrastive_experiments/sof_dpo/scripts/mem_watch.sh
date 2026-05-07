#!/usr/bin/env bash
# Tail-style pod RAM % monitor (reads cgroup, not host).
# The host has 3 TiB but this pod's cgroup is capped at 2 TiB —
# `free` reports against the host total and is therefore misleading.
#
# IMPORTANT: cgroup `memory.current` includes the page cache (file-backed
# pages from reading large videos / checkpoints / parquet). That cache is
# reclaimable and is NOT what triggers OOM. Kubelet's "working set" metric
# (what the OOM killer cares about) is:
#       working_set = memory.current - inactive_file
# We report both numbers so you can tell real usage from cached I/O.
#
# Usage: bash mem_watch.sh [interval_sec]   (default 5s)
INTERVAL="${1:-5}"

# Detect cgroup v2 vs v1
if [[ -r /sys/fs/cgroup/memory.max && -r /sys/fs/cgroup/memory.current ]]; then
    CGV=2
    LIMIT_FILE=/sys/fs/cgroup/memory.max
    USAGE_FILE=/sys/fs/cgroup/memory.current
    STAT_FILE=/sys/fs/cgroup/memory.stat
elif [[ -r /sys/fs/cgroup/memory/memory.limit_in_bytes ]]; then
    CGV=1
    LIMIT_FILE=/sys/fs/cgroup/memory/memory.limit_in_bytes
    USAGE_FILE=/sys/fs/cgroup/memory/memory.usage_in_bytes
    STAT_FILE=/sys/fs/cgroup/memory/memory.stat
else
    echo "❌ no cgroup memory files found"
    exit 1
fi

LIMIT_BYTES=$(cat "$LIMIT_FILE" 2>/dev/null)
# cgroup v2 reports "max" when uncapped — fall back to host MemTotal
if [[ "$LIMIT_BYTES" == "max" || -z "$LIMIT_BYTES" ]]; then
    LIMIT_BYTES=$(awk '/^MemTotal:/ {print $2*1024}' /proc/meminfo)
fi
LIMIT_GIB=$(( LIMIT_BYTES / 1073741824 ))
echo "pod memory limit: ${LIMIT_GIB} GiB  (cgroup v${CGV})"
echo "    WS = working set (anon + active file, OOM-relevant)"
echo "   tot = memory.current (includes reclaimable page cache)"
echo "---"

# Pull a field out of memory.stat by key name
stat_val() {
    awk -v k="$1" '$1==k {print $2; exit}' "$STAT_FILE" 2>/dev/null
}

while true; do
    USED_BYTES=$(cat "$USAGE_FILE" 2>/dev/null || echo 0)
    if [[ "$CGV" == "2" ]]; then
        INACTIVE_FILE=$(stat_val inactive_file)
        FILE_CACHE=$(stat_val file)
        ANON=$(stat_val anon)
    else
        INACTIVE_FILE=$(stat_val total_inactive_file)
        FILE_CACHE=$(stat_val total_cache)
        ANON=$(stat_val total_rss)
    fi
    INACTIVE_FILE=${INACTIVE_FILE:-0}
    FILE_CACHE=${FILE_CACHE:-0}
    ANON=${ANON:-0}

    WS_BYTES=$(( USED_BYTES - INACTIVE_FILE ))
    (( WS_BYTES < 0 )) && WS_BYTES=0

    WS_PCT=$(( WS_BYTES * 100 / LIMIT_BYTES ))
    TOT_PCT=$(( USED_BYTES * 100 / LIMIT_BYTES ))
    WS_GIB=$(( WS_BYTES / 1073741824 ))
    TOT_GIB=$(( USED_BYTES / 1073741824 ))
    ANON_GIB=$(( ANON / 1073741824 ))
    CACHE_GIB=$(( FILE_CACHE / 1073741824 ))

    printf 'WS %3d%% (%4d GiB) | tot %3d%% (%4d GiB) | anon %4d GiB  cache %4d GiB\n' \
        "$WS_PCT" "$WS_GIB" "$TOT_PCT" "$TOT_GIB" "$ANON_GIB" "$CACHE_GIB"
    sleep "$INTERVAL"
done
