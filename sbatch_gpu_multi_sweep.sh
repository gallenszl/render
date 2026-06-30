#!/bin/bash
#SBATCH --job-name=multi_sweep
#SBATCH --partition=gpu
#SBATCH --qos=low
#SBATCH --gres=gpu:h200:4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=200
#SBATCH --mem=750G
#SBATCH --time=01:00:00
#SBATCH --exclude=lrc-alpha-sg-gpu14
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Single 4-GPU allocation sweep over N_PROCS_PER_GPU ∈ {1, 2, 4, 8}.
# Each config runs back-to-back in the same allocation → fair comparison.
# Tests how multi-Blender scaling behaves on 4 GPUs simultaneously (vs the
# single-GPU bench at 9.7). For each N_PROCS: fork 4 × N_PROCS Blender procs,
# pinned to GPUs 0..3 via CUDA_VISIBLE_DEVICES.

set -e
ROOT="$HOME/code/render_pipeline"
BLENDER="$HOME/tools/blender-4.2.9-linux-x64/blender"
BLPY="$HOME/tools/blender-4.2.9-linux-x64/4.2/python/bin/python3.11"

export XDG_CACHE_HOME=$HOME/.cache
export LD_LIBRARY_PATH=$HOME/tools/x11_libs:$LD_LIBRARY_PATH

OBJ_LIST="${OBJ_LIST:-$HOME/tmp/multi_gpu_scaling_test/obj_list_100.txt}"
SWEEP_ROOT="${SWEEP_ROOT:-$HOME/tmp/multi_gpu_scaling_test/sweep_$SLURM_JOB_ID}"
N_GPUS=4   # locked: matches --gres above
N_VALUES="${N_VALUES:-1 2 4 8}"
NUM_VIEWS="${NUM_VIEWS:-40}"
SAMPLES="${SAMPLES:-128}"
RESOLUTION="${RESOLUTION:-512}"

mkdir -p "$SWEEP_ROOT"

SCRATCH_BASE="/tmp/render_${SLURM_JOB_ID}_sweep"
mkdir -p "$SCRATCH_BASE"
trap 'rm -rf "$SCRATCH_BASE"' EXIT

N_OBJS=$(wc -l < "$OBJ_LIST")
TOTAL_VIEWS=$(( N_OBJS * NUM_VIEWS ))

echo "=== sbatch_gpu_multi_sweep (4-GPU, sweep N_PROCS_PER_GPU) ==="
echo "Node: $(hostname)  JobID: $SLURM_JOB_ID"
echo "SWEEP_ROOT=$SWEEP_ROOT"
echo "N_GPUS=$N_GPUS (fixed)  N_VALUES (sweep over N_PROCS_PER_GPU)=$N_VALUES"
echo "OBJ_LIST=$OBJ_LIST ($N_OBJS objs)  NUM_VIEWS=$NUM_VIEWS  TOTAL_VIEWS=$TOTAL_VIEWS"
echo "Slurm gave CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Other jobs on this node:"
squeue -h -w "$(hostname)" -o "  %u %j" 2>&1 || true
echo "===="

nvidia-smi dmon -s u -d 1 > "$SWEEP_ROOT/gpu_util.log" 2>&1 &
DMON_PID=$!

for N_PROCS in $N_VALUES; do
    OUT_N="$SWEEP_ROOT/N${N_PROCS}"
    rm -rf "$OUT_N"
    mkdir -p "$OUT_N" "$OUT_N/_logs"
    TOTAL_PROCS=$((N_GPUS * N_PROCS))
    PROC_CHUNK=$(( (N_OBJS + TOTAL_PROCS - 1) / TOTAL_PROCS ))

    echo
    echo "##### N_PROCS_PER_GPU=$N_PROCS  TOTAL_PROCS=$TOTAL_PROCS  PROC_CHUNK=$PROC_CHUNK #####"

    T0=$(date +%s.%N)
    PIDS=()
    for gpu_idx in $(seq 0 $((N_GPUS - 1))); do
        for proc_idx in $(seq 0 $((N_PROCS - 1))); do
            global_idx=$((gpu_idx * N_PROCS + proc_idx))
            P_START=$((global_idx * PROC_CHUNK + 1))
            P_END=$(((global_idx + 1) * PROC_CHUNK))
            PROC_LIST="$SCRATCH_BASE/N${N_PROCS}_proc${global_idx}.txt"
            sed -n "${P_START},${P_END}p" "$OBJ_LIST" > "$PROC_LIST"
            if [ ! -s "$PROC_LIST" ]; then continue; fi
            SCRATCH_I="$SCRATCH_BASE/N${N_PROCS}_scratch_g${gpu_idx}_p${proc_idx}"
            mkdir -p "$SCRATCH_I"
            PROC_LOG="$OUT_N/_logs/g${gpu_idx}_p${proc_idx}.log"
            (
                export CUDA_VISIBLE_DEVICES=$gpu_idx
                "$BLPY" "$ROOT/render.py" \
                    --obj_list "$PROC_LIST" \
                    --output_dir "$OUT_N" \
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
    WALL=$(echo "$T1 - $T0" | bc)
    N_TARS=$(find "$OUT_N" -maxdepth 1 -name "*.tar" | wc -l)
    VPS=$(echo "scale=4; $TOTAL_VIEWS / $WALL" | bc)
    PER_GPU=$(echo "scale=4; $VPS / $N_GPUS" | bc)
    echo "N_PROCS=$N_PROCS  WALL=${WALL}s  N_TARS=$N_TARS  VPS=$VPS  PER_GPU_VPS=$PER_GPU"
    cat > "$OUT_N/result.json" <<EOF
{
  "n_gpus": $N_GPUS,
  "n_procs_per_gpu": $N_PROCS,
  "total_procs": $TOTAL_PROCS,
  "n_objs": $N_OBJS,
  "num_views": $NUM_VIEWS,
  "total_views": $TOTAL_VIEWS,
  "wall_sec": $WALL,
  "n_tars": $N_TARS,
  "views_per_sec": $VPS,
  "per_gpu_vps": $PER_GPU
}
EOF
done

kill "$DMON_PID" 2>/dev/null || true

echo
echo "=== SWEEP SUMMARY (4 GPUs, varying N_PROCS_PER_GPU) ==="
echo "Node: $(hostname)"
for N in $N_VALUES; do
    R="$SWEEP_ROOT/N${N}/result.json"
    if [ -f "$R" ]; then
        WALL=$("$BLPY" -c "import json; print(f'{json.load(open(\"$R\"))[\"wall_sec\"]:.1f}')")
        VPS=$("$BLPY" -c "import json; print(f'{json.load(open(\"$R\"))[\"views_per_sec\"]:.2f}')")
        PG=$("$BLPY" -c "import json; print(f'{json.load(open(\"$R\"))[\"per_gpu_vps\"]:.2f}')")
        NT=$("$BLPY" -c "import json; print(json.load(open(\"$R\"))[\"n_tars\"])")
        echo "  N_PROCS_PER_GPU=$N  wall=${WALL}s  vps=${VPS}  per_gpu_vps=${PG}  tars=${NT}"
    fi
done
echo "Sweep root: $SWEEP_ROOT"
