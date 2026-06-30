#!/bin/bash
#SBATCH --job-name=rng_smoke_5obj
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -e
ROOT="$HOME/code/render_pipeline"
BLENDER="$HOME/tools/blender-4.2.9-linux-x64/blender"
BLPY="$HOME/tools/blender-4.2.9-linux-x64/4.2/python/bin/python3.11"

export XDG_CACHE_HOME=$HOME/.cache
export LD_LIBRARY_PATH=$HOME/tools/x11_libs:$LD_LIBRARY_PATH

OBJ_LIST="$HOME/tmp/render_bench_100/obj_list_5.txt"
OUTPUT_DIR="$HOME/tmp/render_bench_100/smoke_5_gpu"
mkdir -p "$OUTPUT_DIR"

# C8: /tmp scratch + rsync to final output
SCRATCH="/tmp/render_${SLURM_JOB_ID}_$$"
mkdir -p "$SCRATCH"
trap 'rm -rf "$SCRATCH"' EXIT

nvidia-smi | head -10
echo "===="
t0=$(date +%s)
"$BLPY" $ROOT/render.py \
    --obj_list "$OBJ_LIST" \
    --output_dir "$OUTPUT_DIR" \
    --scratch_dir "$SCRATCH" \
    --num_views 40 \
    --resolution 512 \
    --device GPU \
    --samples 128 \
    --blender "$BLENDER" \
    --persistent
t1=$(date +%s)
echo "===="
echo "TOTAL_WALL_SECONDS=$((t1-t0))"
