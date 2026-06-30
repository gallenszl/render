#!/bin/bash
#SBATCH --job-name=rng_render_cpu
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=15G
#SBATCH --time=04:00:00
#SBATCH --array=0-7%8
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

# Note: 8 array tasks × 8 cores = 64 cores = full user quota.
# Run on the single CPU node lrc-alpha-sg-cpu01.

set -e
ROOT="$HOME/code/render_pipeline"
BLENDER="$HOME/tools/blender-4.2.9-linux-x64/blender"
BLPY="$HOME/tools/blender-4.2.9-linux-x64/4.2/python/bin/python3.11"

export LD_LIBRARY_PATH=$HOME/tools/x11_libs:$LD_LIBRARY_PATH

OBJ_LIST="${OBJ_LIST:?must set OBJ_LIST}"
OUTPUT_DIR="${OUTPUT_DIR:?must set OUTPUT_DIR}"
NUM_VIEWS="${NUM_VIEWS:-40}"
SAMPLES="${SAMPLES:-128}"
RESOLUTION="${RESOLUTION:-512}"

N_OBJS=$(wc -l < "$OBJ_LIST")
NTASKS="${SLURM_ARRAY_TASK_COUNT:-1}"
CHUNK=$(( (N_OBJS + NTASKS - 1) / NTASKS ))
START=$(( SLURM_ARRAY_TASK_ID * CHUNK ))
END=$(( START + CHUNK ))

CHUNK_FILE="/tmp/rng_chunk_cpu_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.txt"
sed -n "$((START+1)),${END}p" "$OBJ_LIST" > "$CHUNK_FILE"
CHUNK_SIZE=$(wc -l < "$CHUNK_FILE")
echo "[task $SLURM_ARRAY_TASK_ID] processing $CHUNK_SIZE objs"

if [ $CHUNK_SIZE -eq 0 ]; then
    exit 0
fi

TASK_OUT="$OUTPUT_DIR"
mkdir -p "$TASK_OUT"

# C8: /tmp scratch + rsync to final output
SCRATCH="/tmp/render_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$SCRATCH"
trap 'rm -rf "$SCRATCH"' EXIT

"$BLPY" $ROOT/render.py \
    --obj_list "$CHUNK_FILE" \
    --output_dir "$TASK_OUT" \
    --scratch_dir "$SCRATCH" \
    --num_views "$NUM_VIEWS" \
    --resolution "$RESOLUTION" \
    --device CPU \
    --samples "$SAMPLES" \
    --blender "$BLENDER" \
    --persistent

echo "[task $SLURM_ARRAY_TASK_ID] done"
