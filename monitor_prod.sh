#!/bin/bash
# Monitor production rendering health. Run periodically.
# Checks: tar count progress, per-fork memory trajectory, any FAILs.
#
# Usage: bash monitor_prod.sh

set -u
ROOT=$HOME/data/objaverse_renders_44798
LOGDIR=$ROOT/_logs

JOBS=$(squeue -h -u $USER -o "%i %j %T %M %D %R" 2>/dev/null | grep rng_part | sort)
echo "=== $(date +'%H:%M:%S') ==="
echo "$JOBS"
echo

N_TARS=$(ls $ROOT/*.tar 2>/dev/null | wc -l)
echo "Total tars: $N_TARS / 44798 ($(echo "scale=2; 100*$N_TARS/44798" | bc)%)"
echo

# Latest [mem] line per fork log (shows RSS trajectory)
echo "=== Per-fork latest [mem] snapshot (RSS should plateau, meshes should be ~1-3) ==="
for log in $(ls -t $LOGDIR/job*.log 2>/dev/null | head -16); do
    name=$(basename $log .log)
    last_mem=$(grep "\[mem\]" "$log" 2>/dev/null | tail -1 | grep -oE "meshes=[0-9]+ lights=[0-9]+ cameras=[0-9]+ images=[0-9]+ rss=[0-9.]+GB")
    obj_count=$(grep -c "\[mem\]" "$log" 2>/dev/null || echo 0)
    if [ -z "$last_mem" ]; then
        rendering=$(tail -1 "$log" 2>/dev/null | head -c 60)
        echo "  $name: not yet finished 1st obj  [tail: $rendering...]"
    else
        echo "  $name: $obj_count obj done | $last_mem"
    fi
done
echo

# Check for any failures
echo "=== Errors / FAILs in fork logs ==="
grep -l "FAILED\|Traceback\|out of memory\|orphans_purge" $LOGDIR/*.log 2>/dev/null | head -5 || echo "(none)"
echo

# Memory health alerts
echo "=== Memory health alerts ==="
for log in $LOGDIR/job*.log; do
    [ -f "$log" ] || continue
    name=$(basename $log .log)
    max_rss=$(grep "\[mem\]" "$log" 2>/dev/null | grep -oE "rss=[0-9.]+GB" | sort -t= -k2 -n | tail -1)
    if [ -n "$max_rss" ]; then
        val=$(echo "$max_rss" | grep -oE "[0-9.]+")
        if (( $(echo "$val > 20" | bc -l) )); then
            echo "  ⚠️  $name: $max_rss (HIGH — possible leak)"
        fi
    fi
done | head -10
echo "(empty = no high-RSS forks)"
