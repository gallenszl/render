#!/bin/bash
#SBATCH --job-name=profile_smoke
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --exclude=lrc-alpha-sg-gpu14
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Per-stage render profile smoke: 5 obj × 10 view in B mode (/tmp scratch).
# Each obj writes phases.json with 4-stage timing (setup/render/post/meta).

set -e
ROOT="$HOME/code/render_pipeline"
BLENDER="$HOME/tools/blender-4.2.9-linux-x64/blender"
BLPY="$HOME/tools/blender-4.2.9-linux-x64/4.2/python/bin/python3.11"

export XDG_CACHE_HOME=$HOME/.cache
export LD_LIBRARY_PATH=$HOME/tools/x11_libs:$LD_LIBRARY_PATH

OBJ_LIST="$HOME/tmp/render_bench_100/obj_list_5.txt"
OUTPUT_DIR="$HOME/tmp/render_bench_100/profile_smoke_5x10"
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# C8: /tmp scratch + rsync
SCRATCH="/tmp/render_${SLURM_JOB_ID}_$$"
mkdir -p "$SCRATCH"
trap 'rm -rf "$SCRATCH"' EXIT

echo "=== profile_smoke ==="
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "===="

T0=$(date +%s.%N)
"$BLPY" "$ROOT/render.py" \
    --obj_list "$OBJ_LIST" \
    --output_dir "$OUTPUT_DIR" \
    --scratch_dir "$SCRATCH" \
    --num_views 10 \
    --resolution 512 --device GPU --samples 128 \
    --offset_max 0.15 --offset_zero_prob 0.20 \
    --radius_min_factor 0.95 --radius_max_factor 1.15 \
    --fov_min_deg 25 --fov_max_deg 70 \
    --blender "$BLENDER" --persistent
T1=$(date +%s.%N)
echo "===="
echo "TOTAL_WALL=$(echo "$T1 - $T0" | bc)"

# Aggregate phases.json across all objs
"$BLPY" - <<EOF
import glob, json, os, statistics
files = sorted(glob.glob("$OUTPUT_DIR/*/phases.json"))
print(f"\\nAggregating {len(files)} phases.json files...")
all_views = []
for f in files:
    d = json.load(open(f))
    all_views.extend(d["per_view"])
print(f"Total views: {len(all_views)}")
print(f"\\n{'Stage':<8} {'Mean':>10} {'Median':>10} {'Max':>10} {'Total':>10} {'% of total':>12}")
total_sum = sum(v["total"] for v in all_views)
for k in ("setup", "render", "post", "meta", "total"):
    vals = [v[k] for v in all_views]
    s = sum(vals)
    pct = 100*s/total_sum if k != "total" else 100.0
    print(f"{k:<8} {statistics.mean(vals):>10.4f} {statistics.median(vals):>10.4f} {max(vals):>10.4f} {s:>10.3f} {pct:>11.1f}%")
EOF
