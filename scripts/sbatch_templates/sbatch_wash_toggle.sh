#!/bin/bash
#SBATCH --job-name=wash-toggle
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --exclude=lrc-alpha-sg-gpu05,lrc-alpha-sg-gpu06,lrc-alpha-sg-gpu14
#SBATCH --output=/home/z50057756/tmp/wash_fix_bisect/sbatch_repro_%j.out
#SBATCH --error=/home/z50057756/tmp/wash_fix_bisect/sbatch_repro_%j.err

set -euo pipefail
OLDHOME=/mnt/data-alpha-sg-01/team-camera/home/z50057756
BLENDER=$OLDHOME/tools/blender-4.2.9-linux-x64/blender

export XDG_CACHE_HOME=$OLDHOME/.cache
export LD_LIBRARY_PATH=$OLDHOME/tools/x11_libs:${LD_LIBRARY_PATH:-}
export OPTIX_CACHE_PATH=$OLDHOME/.cache/OptixCache

cd /home/z50057756/tmp/phase2_smoke
"$BLENDER" --background --python-use-system-env --python wash_toggle_test.py
