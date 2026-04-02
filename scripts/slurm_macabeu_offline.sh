#!/bin/bash
# SLURM array job: MACABEU offline (frozen policy) evaluation with BLEURT
# 4 tasks × 4 attackers = 16 jobs
#
# Prerequisites: macabeu repo cloned on HPC, trained models in models/
# Submit with: cd ~/macabeu && sbatch scripts/slurm_macabeu_offline.sh

#SBATCH -J mac_off
#SBATCH -p medium
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -c 4
#SBATCH --array=0-15
#SBATCH -o logs/macabeu_offline_%A_%a.out
#SBATCH -e logs/macabeu_offline_%A_%a.err

TASKS=(PR2 FC HN RD)
ATTACKERS=(DeepWordBug BERTattack PWWS Genetic)

i=$SLURM_ARRAY_TASK_ID
TASK=${TASKS[$((i / 4))]}
ATTACK=${ATTACKERS[$((i % 4))]}

VICTIM="BiLSTM"
DATA_PATH="../BODEGA/data/$TASK"
MODEL_PATH="../BODEGA/data/$TASK/BiLSTM-512.pth"
POLICY_PATH="models/${TASK}_BiLSTM.pth"
OUT_DIR="results/offline"

module load Miniconda3
eval "$(conda shell.bash hook)"
conda activate bodega
export PYTHONPATH="../BODEGA:."
mkdir -p "$OUT_DIR" logs

echo "[$i] MACABEU OFFLINE | $TASK | $ATTACK | $VICTIM"

python runs/eval_defense_agent.py \
    "$TASK" "$VICTIM" "$DATA_PATH" "$MODEL_PATH" "$POLICY_PATH" "$OUT_DIR" \
    --attackers "$ATTACK" \
    --semantic_scorer BLEURT
