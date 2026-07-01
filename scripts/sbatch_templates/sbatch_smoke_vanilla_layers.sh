#!/bin/bash
#SBATCH --job-name=smoke-vanilla_layers
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=100G
#SBATCH --time=01:00:00
#SBATCH --exclude=lrc-alpha-sg-gpu05,lrc-alpha-sg-gpu06,lrc-alpha-sg-gpu14
#SBATCH --output=/home/z50057756/tmp/phase2_smoke/sbatch_vanilla_layers_%j.out
#SBATCH --error=/home/z50057756/tmp/phase2_smoke/sbatch_vanilla_layers_%j.err

set -euo pipefail
OLDHOME=/mnt/data-alpha-sg-01/team-camera/home/z50057756
ROOT=/home/z50057756/code/render_pipeline
BLENDER=$OLDHOME/tools/blender-4.2.9-linux-x64/blender
BLPY=$OLDHOME/tools/blender-4.2.9-linux-x64/4.2/python/bin/python3.11

export XDG_CACHE_HOME=$OLDHOME/.cache
export LD_LIBRARY_PATH=$OLDHOME/tools/x11_libs:${LD_LIBRARY_PATH:-}
export OPTIX_CACHE_PATH=$OLDHOME/.cache/OptixCache
mkdir -p $OPTIX_CACHE_PATH
export R_FRAME=0.62
export ENABLE_STRATEGY_E=1

OBJ_LIST=/home/z50057756/tmp/phase2_smoke/obj_list_100.txt
OUTPUT_DIR=/home/z50057756/tmp/phase2_smoke/render_out_vanilla_layers
mkdir -p "$OUTPUT_DIR/_logs"
N_OBJS=$(wc -l < "$OBJ_LIST")
PROC_CHUNK=$(( (N_OBJS + 4 - 1) / 4 ))
SCRATCH_BASE="/tmp/smoke_vanilla_layers_${SLURM_JOB_ID}"
mkdir -p "$SCRATCH_BASE"
trap 'rm -rf "$SCRATCH_BASE"' EXIT

echo "MAD_N=30, R_FRAME=0.62"
PIDS=()
for proc_idx in 0 1 2 3; do
    P_START=$((proc_idx * PROC_CHUNK + 1))
    P_END=$(((proc_idx + 1) * PROC_CHUNK))
    PROC_LIST="$SCRATCH_BASE/proc${proc_idx}.txt"
    sed -n "${P_START},${P_END}p" "$OBJ_LIST" > "$PROC_LIST"
    [ -s "$PROC_LIST" ] || continue
    SCRATCH_I="$SCRATCH_BASE/scratch_p${proc_idx}"
    mkdir -p "$SCRATCH_I"
    PROC_LOG="$OUTPUT_DIR/_logs/job${SLURM_JOB_ID}_p${proc_idx}.log"
    (
        export CUDA_VISIBLE_DEVICES=0
        export R_FRAME=0.62
        export ENABLE_STRATEGY_E=1
        "$BLPY" "$ROOT/render.py" --obj_list "$PROC_LIST"             --output_dir "$OUTPUT_DIR" --scratch_dir "$SCRATCH_I"             --tar_output --num_views 40 --resolution 512 --device GPU             --samples 128 --blender "$BLENDER" --persistent --r_frame 0.62             > "$PROC_LOG" 2>&1
    ) &
    PIDS+=($!)
done
for pid in "${PIDS[@]}"; do wait "$pid"; done
echo "tars: $(ls $OUTPUT_DIR/*.tar 2>/dev/null | wc -l) quar: $([ -f $OUTPUT_DIR/quarantine.txt ] && wc -l < $OUTPUT_DIR/quarantine.txt || echo 0)"
