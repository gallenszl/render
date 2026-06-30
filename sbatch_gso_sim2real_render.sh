#!/bin/bash
#SBATCH --job-name=rng_gso_sim2real_render
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:h200:4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=400G
#SBATCH --time=02:00:00
#SBATCH --exclude=lrc-alpha-sg-gpu05,lrc-alpha-sg-gpu06
#SBATCH --output=/home/z50057756/code/render_pipeline/slurm_logs/%x_%j.out
#SBATCH --error=/home/z50057756/code/render_pipeline/slurm_logs/%x_%j.err

# Sim2real GSO re-render — matches production 44,798 params (plan §1.3 / §8.7.3).
# Output: loose files <scene>/000.png ... 024.png + 000_depth.png ... 024_depth.png + transforms.json
# NOT tar (dataset_gso_ours.py expects loose; use_tar=False hard-coded).

set -euo pipefail
mkdir -p /home/z50057756/code/render_pipeline/slurm_logs

# === post-migration absolute paths (tools/cache still on /mnt/data-alpha-sg-01) ===
OLDHOME=/mnt/data-alpha-sg-01/team-camera/home/z50057756
ROOT=/home/z50057756/code/render_pipeline
BLENDER=$OLDHOME/tools/blender-4.2.9-linux-x64/blender
BLPY=$OLDHOME/tools/blender-4.2.9-linux-x64/4.2/python/bin/python3.11

export XDG_CACHE_HOME=$OLDHOME/.cache
export LD_LIBRARY_PATH=$OLDHOME/tools/x11_libs:${LD_LIBRARY_PATH:-}
export OPTIX_CACHE_PATH=$OLDHOME/.cache/OptixCache
mkdir -p $OPTIX_CACHE_PATH

# === inputs ===
OBJ_LIST=${OBJ_LIST:-/home/z50057756/data/gso_meshes/gso_obj_list.txt}
OUTPUT_DIR=${OUTPUT_DIR:-/home/z50057756/data/gso_sim2real_25v}
NUM_VIEWS=25
SAMPLES=128
RESOLUTION=512
N_GPUS=4
N_PROCS_PER_GPU=4
TOTAL_PROCS=$((N_GPUS * N_PROCS_PER_GPU))

# Sim2real production params (plan §1.3 / §8.7.3 — same as 44,798 production)
A1_OFFSET=0.15
A1_ZERO_PROB=0.20
FOV_MIN=25
FOV_MAX=70
R2_MIN=0.95
R2_MAX=1.15
R_FRAME=0.684

mkdir -p $OUTPUT_DIR $OUTPUT_DIR/_logs
N_OBJS=$(wc -l < $OBJ_LIST)
PROC_CHUNK=$(( (N_OBJS + TOTAL_PROCS - 1) / TOTAL_PROCS ))

echo "=== Sim2real GSO render ==="
echo "Job:        $SLURM_JOB_ID"
echo "Node:       $(hostname)"
echo "OBJ_LIST:   $OBJ_LIST  ($N_OBJS objs)"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "Views: $NUM_VIEWS  Res: $RESOLUTION  Samples: $SAMPLES"
echo "Sim2real:  FoV [$FOV_MIN, $FOV_MAX]°  R2 [$R2_MIN, $R2_MAX]  A1 offset $A1_OFFSET  r_frame $R_FRAME"
echo "Procs:     $N_GPUS GPU × $N_PROCS_PER_GPU per-GPU = $TOTAL_PROCS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "===="

SCRATCH_BASE=/tmp/render_${SLURM_JOB_ID}
mkdir -p $SCRATCH_BASE
trap 'rm -rf $SCRATCH_BASE' EXIT

# Monitor GPU util
nvidia-smi dmon -s u -d 5 > $OUTPUT_DIR/_logs/gpu_util_${SLURM_JOB_ID}.log 2>&1 &
DMON_PID=$!

T0=$(date +%s.%N)
PIDS=()
for gpu_idx in $(seq 0 $((N_GPUS - 1))); do
  for proc_idx in $(seq 0 $((N_PROCS_PER_GPU - 1))); do
    global_idx=$((gpu_idx * N_PROCS_PER_GPU + proc_idx))
    P_START=$((global_idx * PROC_CHUNK + 1))
    P_END=$(((global_idx + 1) * PROC_CHUNK))
    PROC_LIST=$SCRATCH_BASE/proc${global_idx}.txt
    sed -n "${P_START},${P_END}p" $OBJ_LIST > $PROC_LIST
    if [ ! -s $PROC_LIST ]; then continue; fi
    SCRATCH_I=$SCRATCH_BASE/scratch_g${gpu_idx}_p${proc_idx}
    mkdir -p $SCRATCH_I
    PROC_LOG=$OUTPUT_DIR/_logs/job${SLURM_JOB_ID}_g${gpu_idx}_p${proc_idx}.log
    (
      export CUDA_VISIBLE_DEVICES=$gpu_idx
      $BLPY $ROOT/render.py \
        --obj_list $PROC_LIST \
        --output_dir $OUTPUT_DIR \
        --scratch_dir $SCRATCH_I \
        --num_views $NUM_VIEWS \
        --resolution $RESOLUTION \
        --device GPU \
        --samples $SAMPLES \
        --offset_max $A1_OFFSET \
        --offset_zero_prob $A1_ZERO_PROB \
        --fov_min_deg $FOV_MIN \
        --fov_max_deg $FOV_MAX \
        --radius_min_factor $R2_MIN \
        --radius_max_factor $R2_MAX \
        --r_frame $R_FRAME \
        --blender $BLENDER \
        --persistent \
        > $PROC_LOG 2>&1
    ) &
    PIDS+=($!)
  done
done

for pid in "${PIDS[@]}"; do wait $pid; done

T1=$(date +%s.%N)
TOTAL_WALL=$(echo "$T1 - $T0" | bc)
kill $DMON_PID 2>/dev/null || true

N_DIRS=$(find $OUTPUT_DIR -maxdepth 1 -mindepth 1 -type d ! -name "_logs" | wc -l)
echo "===="
echo "TOTAL_WALL: ${TOTAL_WALL}s"
echo "OUTPUT scenes: $N_DIRS"
echo "VIEWS/SEC: $(echo "scale=2; $N_DIRS * $NUM_VIEWS / $TOTAL_WALL" | bc)"
