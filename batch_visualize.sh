#!/bin/bash
LOG=/home/boris/workspace/HOMIE-toolkit/batch_visualize.log
VENV=/home/boris/workspace/HOMIE-toolkit/.venv/bin/python
SCRIPT=/home/boris/workspace/HOMIE-toolkit/examples/example_visualize_rrd.py
DATA=/home/boris/workspace/HOMIE-toolkit/data/xperience-10m

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

TOTAL=0
DONE=0
FAIL=0
SKIP=0

for ann in $(find "$DATA" -name "annotation.hdf5" -type f | sort); do
    ep_dir=$(dirname "$ann")
    rrd="$ep_dir/visualization.rrd"
    rel=${ep_dir#$DATA/}
    TOTAL=$((TOTAL+1))

    if [ -f "$rrd" ]; then
        SKIP=$((SKIP+1))
        continue
    fi

    log "[$((DONE+FAIL+1))/$((TOTAL))] Generating: $rel"
    cd /home/boris/workspace/HOMIE-toolkit
    $VENV "$SCRIPT" --data_root "$ep_dir" --output_rrd visualization.rrd >> "$LOG" 2>&1
    rc=$?
    if [ $rc -eq 0 ] && [ -f "$rrd" ]; then
        DONE=$((DONE+1))
        log "OK: $rel"
    else
        FAIL=$((FAIL+1))
        log "FAIL (rc=$rc): $rel"
    fi
done

log "=== BATCH DONE: total=$TOTAL skip=$SKIP done=$DONE fail=$FAIL ==="
