#!/bin/bash
#SBATCH --job-name=rng_render_gpu
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --array=0%1
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

set -e
ROOT="$HOME/code/render_pipeline"
BLENDER="$HOME/tools/blender-4.2.9-linux-x64/blender"
BLPY="$HOME/tools/blender-4.2.9-linux-x64/4.2/python/bin/python3.11"

# Engineering optimizations (Tier C):
export XDG_CACHE_HOME=$HOME/.cache                      # C2: persist OPTIX kernels on JuiceFS
export LD_LIBRARY_PATH=$HOME/tools/x11_libs:$LD_LIBRARY_PATH

# Args (set externally before submit):
#   OBJ_LIST   - text file with one .glb path per line
#   OUTPUT_DIR - base output dir
#   NUM_VIEWS  - per-object views (default 40)
OBJ_LIST="${OBJ_LIST:?must set OBJ_LIST}"
OUTPUT_DIR="${OUTPUT_DIR:?must set OUTPUT_DIR}"
NUM_VIEWS="${NUM_VIEWS:-40}"
SAMPLES="${SAMPLES:-128}"
RESOLUTION="${RESOLUTION:-512}"

# Slice the obj list among the actual array tasks
N_OBJS=$(wc -l < "$OBJ_LIST")
NTASKS="${SLURM_ARRAY_TASK_COUNT:-1}"
CHUNK=$(( (N_OBJS + NTASKS - 1) / NTASKS ))
START=$(( SLURM_ARRAY_TASK_ID * CHUNK ))
END=$(( START + CHUNK ))

CHUNK_FILE="/tmp/rng_chunk_gpu_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.txt"
sed -n "$((START+1)),${END}p" "$OBJ_LIST" > "$CHUNK_FILE"
CHUNK_SIZE=$(wc -l < "$CHUNK_FILE")
echo "[task $SLURM_ARRAY_TASK_ID] processing $CHUNK_SIZE objs from $OBJ_LIST line $((START+1))-$END"

if [ $CHUNK_SIZE -eq 0 ]; then
    echo "[task $SLURM_ARRAY_TASK_ID] empty chunk, exiting"
    exit 0
fi

# All tasks write into the SAME output dir; different uids per chunk so no collision.
TASK_OUT="$OUTPUT_DIR"
mkdir -p "$TASK_OUT"

# C8: /tmp scratch + rsync to final output (~17-29% faster than direct JuiceFS)
SCRATCH="/tmp/render_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$SCRATCH"
trap 'rm -rf "$SCRATCH"' EXIT

# Run orchestrator in persistent mode (1 Blender, loops over chunk)
nvidia-smi | head -10
"$BLPY" $ROOT/render.py \
    --obj_list "$CHUNK_FILE" \
    --output_dir "$TASK_OUT" \
    --scratch_dir "$SCRATCH" \
    --num_views "$NUM_VIEWS" \
    --resolution "$RESOLUTION" \
    --device GPU \
    --samples "$SAMPLES" \
    --blender "$BLENDER" \
    --persistent

echo "[task $SLURM_ARRAY_TASK_ID] done"
