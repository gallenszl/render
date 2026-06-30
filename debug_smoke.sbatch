#!/bin/bash
#SBATCH --job-name=rng_debug_overlay
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --gres=gpu:h200:1
#SBATCH --exclude=lrc-alpha-sg-gpu14
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -e
ROOT="$HOME/code/render_pipeline"
BLENDER="$HOME/tools/blender-4.2.9-linux-x64/blender"
BLPY="$HOME/tools/blender-4.2.9-linux-x64/4.2/python/bin/python3.11"

export XDG_CACHE_HOME=$HOME/.cache
export LD_LIBRARY_PATH=$HOME/tools/x11_libs:$LD_LIBRARY_PATH

OBJ_LIST="$HOME/tmp/render_bench_100/obj_list_5.txt"
OUTPUT_DIR="$HOME/tmp/render_bench_100/debug_overlay_5x10"
rm -rf "$OUTPUT_DIR"
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
    --num_views 10 \
    --resolution 512 \
    --device GPU \
    --samples 128 \
    --offset_max 0.15 \
    --offset_zero_prob 0.20 \
    --radius_min_factor 0.95 \
    --radius_max_factor 1.15 \
    --blender "$BLENDER" \
    --persistent \
    --debug_overlay
t1=$(date +%s)
echo "===="
echo "TOTAL_WALL_SECONDS=$((t1-t0))"
