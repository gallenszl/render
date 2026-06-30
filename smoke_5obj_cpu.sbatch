#!/bin/bash
#SBATCH --job-name=rng_smoke_cpu
#SBATCH --partition=cpu
#SBATCH --qos=low
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -e
ROOT="$HOME/code/render_pipeline"
BLENDER="$HOME/tools/blender-4.2.9-linux-x64/blender"
BLPY="$HOME/tools/blender-4.2.9-linux-x64/4.2/python/bin/python3.11"

export LD_LIBRARY_PATH=$HOME/tools/x11_libs:$LD_LIBRARY_PATH

OBJ_LIST="$HOME/tmp/render_bench_100/obj_list_5.txt"
OUTPUT_DIR="$HOME/tmp/render_bench_100/smoke_5_gpu"
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

echo "host: $(hostname)"
echo "===="
t0=$(date +%s)
"$BLPY" $ROOT/render.py \
    --obj_list "$OBJ_LIST" \
    --output_dir "$OUTPUT_DIR" \
    --num_views 40 \
    --resolution 512 \
    --device CPU \
    --samples 128 \
    --offset_max 0.25 \
    --radius_min_factor 0.95 \
    --radius_max_factor 1.15 \
    --blender "$BLENDER" \
    --persistent
t1=$(date +%s)
echo "===="
echo "TOTAL_WALL_SECONDS=$((t1-t0))"
