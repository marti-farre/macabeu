#!/bin/bash
# SLURM array job: Clean accuracy evaluation for MACABEU RL defense
# 4 tasks (BiLSTM only for now)
#
# Submit with: cd ~/macabeu && sbatch scripts/slurm_clean_accuracy.sh

#SBATCH -J mac_acc
#SBATCH -p medium
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH --array=0-3
#SBATCH -o logs/mac_clean_acc_%A_%a.out
#SBATCH -e logs/mac_clean_acc_%A_%a.err

TASKS=(PR2 FC HN RD)

i=$SLURM_ARRAY_TASK_ID
TASK=${TASKS[$i]}

VICTIM="BiLSTM"
DATA_PATH="../BODEGA/data/$TASK"
MODEL_PATH="../BODEGA/data/$TASK/BiLSTM-512.pth"
POLICY_PATH="models/${TASK}_${VICTIM}.pth"
OUT_DIR="results/clean_accuracy"

module load Miniconda3
eval "$(conda shell.bash hook)"
conda activate bodega
export PYTHONPATH="../BODEGA:."
mkdir -p "$OUT_DIR" logs

echo "[$i] MACABEU Clean Accuracy | $TASK | $VICTIM"

python runs/eval_clean_accuracy.py \
    "$TASK" "$VICTIM" "$DATA_PATH" "$MODEL_PATH" "$POLICY_PATH" "$OUT_DIR"
