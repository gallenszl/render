#!/bin/bash
#SBATCH --job-name=smoke-2000-fix
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:h200:4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=400G
#SBATCH --time=02:30:00
#SBATCH --exclude=lrc-alpha-sg-gpu05,lrc-alpha-sg-gpu06,lrc-alpha-sg-gpu14
#SBATCH --output=/home/z50057756/tmp/phase2_smoke/sbatch_2000_%j.out
#SBATCH --error=/home/z50057756/tmp/phase2_smoke/sbatch_2000_%j.err

# 2000-obj smoke with current vanilla+Layers pipeline.
# 4 H200 × N=4 procs/GPU = 16 procs (§4.2 benchmark: 29.32 VPS).
# Expected wall: 2000 × 40 / 29.32 ≈ 45 min.

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

OBJ_LIST=/home/z50057756/tmp/phase2_smoke/obj_list_2000.txt
OUTPUT_DIR=/home/z50057756/tmp/phase2_smoke/render_out_2000_fix2
N_GPUS=4
N_PROCS_PER_GPU=4
TOTAL_PROCS=$((N_GPUS * N_PROCS_PER_GPU))

mkdir -p "$OUTPUT_DIR" "$OUTPUT_DIR/_logs"
echo "=== 2000-obj smoke (vanilla+Layers, r_frame=0.62) ==="
echo "Node: $(hostname)  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

N_OBJS=$(wc -l < "$OBJ_LIST")
PROC_CHUNK=$(( (N_OBJS + TOTAL_PROCS - 1) / TOTAL_PROCS ))
SCRATCH_BASE="/tmp/smoke_2000_${SLURM_JOB_ID}"
mkdir -p "$SCRATCH_BASE"
trap 'rm -rf "$SCRATCH_BASE"' EXIT

T0=$(date +%s.%N)

PIDS=()
for gpu_idx in $(seq 0 $((N_GPUS - 1))); do
    for proc_idx in $(seq 0 $((N_PROCS_PER_GPU - 1))); do
        global_idx=$((gpu_idx * N_PROCS_PER_GPU + proc_idx))
        P_START=$((global_idx * PROC_CHUNK + 1))
        P_END=$(((global_idx + 1) * PROC_CHUNK))
        PROC_LIST="$SCRATCH_BASE/proc${global_idx}.txt"
        sed -n "${P_START},${P_END}p" "$OBJ_LIST" > "$PROC_LIST"
        [ -s "$PROC_LIST" ] || continue
        SCRATCH_I="$SCRATCH_BASE/scratch_g${gpu_idx}_p${proc_idx}"
        mkdir -p "$SCRATCH_I"
        PROC_LOG="$OUTPUT_DIR/_logs/job${SLURM_JOB_ID}_g${gpu_idx}_p${proc_idx}.log"
        (
            export CUDA_VISIBLE_DEVICES=$gpu_idx
            export R_FRAME=0.62
            export ENABLE_STRATEGY_E=1
            "$BLPY" "$ROOT/render.py" \
                --obj_list "$PROC_LIST" \
                --output_dir "$OUTPUT_DIR" \
                --scratch_dir "$SCRATCH_I" \
                --tar_output \
                --num_views 40 \
                --resolution 512 \
                --device GPU \
                --samples 128 \
                --blender "$BLENDER" \
                --persistent \
                --r_frame 0.62 \
                > "$PROC_LOG" 2>&1
        ) &
        PIDS+=($!)
    done
done
for pid in "${PIDS[@]}"; do wait "$pid"; done

T1=$(date +%s.%N)
WALL=$(echo "$T1 - $T0" | bc)
N_TARS=$(ls $OUTPUT_DIR/*.tar 2>/dev/null | wc -l)
N_QUAR=$([ -f $OUTPUT_DIR/quarantine.txt ] && wc -l < $OUTPUT_DIR/quarantine.txt || echo 0)
echo
echo "=== 2000-smoke done in ${WALL}s ==="
echo "tars: $N_TARS  quar: $N_QUAR  total_views: $((N_TARS * 40))"
VPS=$(echo "scale=2; $N_TARS * 40 / $WALL" | bc)
echo "throughput: $VPS views/sec"
