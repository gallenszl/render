#!/bin/bash
#SBATCH --job-name=rng_render_multi
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --gres=gpu:h200:4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=400G
#SBATCH --time=01:00:00
#SBATCH --exclude=lrc-alpha-sg-gpu14
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Multi-GPU single-allocation render. Forks N_GPUS x N_PROCS_PER_GPU Blender
# processes, each pinned to one GPU via CUDA_VISIBLE_DEVICES. Each Blender
# writes to its own /tmp scratch then tars + moves to OUTPUT_DIR.
#
# Required env vars:
#   OBJ_LIST   - text file with one .glb path per line
#   OUTPUT_DIR - where <uid>.tar files go
# Optional:
#   NUM_VIEWS=40, RESOLUTION=512, SAMPLES=128
#   N_GPUS=4 (override sbatch --gres accordingly)
#   N_PROCS_PER_GPU=8

set -e
ROOT="$HOME/code/render_pipeline"
BLENDER="$HOME/tools/blender-4.2.9-linux-x64/blender"
BLPY="$HOME/tools/blender-4.2.9-linux-x64/4.2/python/bin/python3.11"

export XDG_CACHE_HOME=$HOME/.cache
export LD_LIBRARY_PATH=$HOME/tools/x11_libs:$LD_LIBRARY_PATH
# R6: persist OPTIX pipeline JIT cache across Blender processes / jobs (cross-node via JuiceFS)
export OPTIX_CACHE_PATH=$HOME/.cache/OptixCache
mkdir -p $OPTIX_CACHE_PATH

OBJ_LIST="${OBJ_LIST:?must set OBJ_LIST}"
OUTPUT_DIR="${OUTPUT_DIR:?must set OUTPUT_DIR}"
NUM_VIEWS="${NUM_VIEWS:-40}"
SAMPLES="${SAMPLES:-128}"
RESOLUTION="${RESOLUTION:-512}"
N_GPUS="${N_GPUS:-4}"
N_PROCS_PER_GPU="${N_PROCS_PER_GPU:-8}"
TOTAL_PROCS=$((N_GPUS * N_PROCS_PER_GPU))

# M1: validate N_GPUS matches Slurm-allocated GPUs (count CUDA_VISIBLE_DEVICES csv)
N_GPUS_ALLOCATED=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
if [ "$N_GPUS" != "$N_GPUS_ALLOCATED" ]; then
    echo "ERROR: N_GPUS=$N_GPUS != Slurm-allocated $N_GPUS_ALLOCATED."
    echo "  Did you forget to pass --gres=gpu:h200:$N_GPUS at sbatch submit time?"
    exit 1
fi

mkdir -p "$OUTPUT_DIR" "$OUTPUT_DIR/_logs"

echo "=== rng_render_multi ==="
echo "Node: $(hostname)"
echo "JobID: $SLURM_JOB_ID"
echo "N_GPUS=$N_GPUS  N_PROCS_PER_GPU=$N_PROCS_PER_GPU  TOTAL_PROCS=$TOTAL_PROCS"
echo "OBJ_LIST=$OBJ_LIST  ($(wc -l < "$OBJ_LIST") objs)"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "NUM_VIEWS=$NUM_VIEWS  RESOLUTION=$RESOLUTION  SAMPLES=$SAMPLES"
echo "CUDA_VISIBLE_DEVICES from Slurm: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Other jobs on this node:"
squeue -h -w "$(hostname)" -o "  %u %j" 2>&1 || true
echo "===="

# Slice OBJ_LIST into TOTAL_PROCS chunks
N_OBJS=$(wc -l < "$OBJ_LIST")
PROC_CHUNK=$(( (N_OBJS + TOTAL_PROCS - 1) / TOTAL_PROCS ))

SCRATCH_BASE="/tmp/render_${SLURM_JOB_ID}"
mkdir -p "$SCRATCH_BASE"
trap 'rm -rf "$SCRATCH_BASE"' EXIT

# M2: snapshot tar count before run so we can compute NEW renders, not total
N_TARS_BEFORE=$(find "$OUTPUT_DIR" -maxdepth 1 -name "*.tar" 2>/dev/null | wc -l)

# GPU monitor (~1Hz)
nvidia-smi dmon -s u -d 1 > "$OUTPUT_DIR/_logs/gpu_util_${SLURM_JOB_ID}.log" 2>&1 &
DMON_PID=$!

T0=$(date +%s.%N)

PIDS=()
for gpu_idx in $(seq 0 $((N_GPUS - 1))); do
    for proc_idx in $(seq 0 $((N_PROCS_PER_GPU - 1))); do
        global_idx=$((gpu_idx * N_PROCS_PER_GPU + proc_idx))
        P_START=$((global_idx * PROC_CHUNK + 1))
        P_END=$(((global_idx + 1) * PROC_CHUNK))
        PROC_LIST="$SCRATCH_BASE/proc${global_idx}.txt"
        sed -n "${P_START},${P_END}p" "$OBJ_LIST" > "$PROC_LIST"
        if [ ! -s "$PROC_LIST" ]; then continue; fi
        SCRATCH_I="$SCRATCH_BASE/scratch_g${gpu_idx}_p${proc_idx}"
        mkdir -p "$SCRATCH_I"
        PROC_LOG="$OUTPUT_DIR/_logs/job${SLURM_JOB_ID}_g${gpu_idx}_p${proc_idx}.log"
        (
            export CUDA_VISIBLE_DEVICES=$gpu_idx
            "$BLPY" "$ROOT/render.py" \
                --obj_list "$PROC_LIST" \
                --output_dir "$OUTPUT_DIR" \
                --scratch_dir "$SCRATCH_I" \
                --tar_output \
                --num_views "$NUM_VIEWS" \
                --resolution "$RESOLUTION" \
                --device GPU \
                --samples "$SAMPLES" \
                --blender "$BLENDER" \
                --persistent \
                > "$PROC_LOG" 2>&1
        ) &
        PIDS+=($!)
    done
done

for pid in "${PIDS[@]}"; do wait "$pid"; done

T1=$(date +%s.%N)
TOTAL_WALL=$(echo "$T1 - $T0" | bc)
kill "$DMON_PID" 2>/dev/null || true

N_TARS_AFTER=$(find "$OUTPUT_DIR" -maxdepth 1 -name "*.tar" 2>/dev/null | wc -l)
# M2: vps based on NEW tars only (skip-if-exists may have skipped many)
N_NEW=$(( N_TARS_AFTER - N_TARS_BEFORE ))
NEW_VIEWS=$(( N_NEW * NUM_VIEWS ))
TOTAL_VIEWS=$(( N_OBJS * NUM_VIEWS ))   # input obj count × views (for reference)
VPS=$(echo "scale=4; $NEW_VIEWS / $TOTAL_WALL" | bc)
PER_GPU_VPS=$(echo "scale=4; $VPS / $N_GPUS" | bc)
AVG_SM=$(awk 'NR>2 && $2 ~ /^[0-9]+$/ {sum+=$2; n++} END {if(n>0) printf "%.1f", sum/n; else print 0}' \
    "$OUTPUT_DIR/_logs/gpu_util_${SLURM_JOB_ID}.log" 2>/dev/null || echo 0)

echo "===="
echo "TOTAL_WALL=${TOTAL_WALL}s"
echo "TARS_BEFORE=${N_TARS_BEFORE}  TARS_AFTER=${N_TARS_AFTER}  NEW_RENDERED=${N_NEW}"
echo "NEW_VIEWS=${NEW_VIEWS}  VIEWS_PER_SEC=${VPS}"
echo "PER_GPU_VPS=${PER_GPU_VPS}"
echo "GPU_AVG_SM_UTIL=${AVG_SM}%"

cat > "$OUTPUT_DIR/result_${SLURM_JOB_ID}.json" <<EOF
{
  "job_id": $SLURM_JOB_ID,
  "node": "$(hostname)",
  "n_gpus": $N_GPUS,
  "n_procs_per_gpu": $N_PROCS_PER_GPU,
  "total_procs": $TOTAL_PROCS,
  "obj_list": "$OBJ_LIST",
  "n_objs_in_list": $N_OBJS,
  "num_views": $NUM_VIEWS,
  "tars_before": $N_TARS_BEFORE,
  "tars_after": $N_TARS_AFTER,
  "new_rendered": $N_NEW,
  "new_views": $NEW_VIEWS,
  "total_wall_sec": $TOTAL_WALL,
  "views_per_sec": $VPS,
  "per_gpu_vps": $PER_GPU_VPS,
  "gpu_avg_sm_util_pct": ${AVG_SM:-0}
}
EOF
echo "Saved $OUTPUT_DIR/result_${SLURM_JOB_ID}.json"
