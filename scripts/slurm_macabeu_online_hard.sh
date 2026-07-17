#!/bin/bash
# SLURM array: standard attackers vs TRUE-ONLINE MACABEU (mv7/HARD reward).
# Ablation for reward-scale question. Skips GEMMA (too expensive).
# 32 cells = 4 tasks × 2 victims (BiLSTM, BERT) × 4 attackers.
#
# Submit:
#   sbatch scripts/slurm_macabeu_online_hard.sh

#SBATCH -J mac_on_hard
#SBATCH -p high
#SBATCH --gres=gpu:1
#SBATCH --constraint=cuda
#SBATCH --mem=16G
#SBATCH -c 4
#SBATCH --array=0-31
#SBATCH -o logs/mac_on_hard_%A_%a.out
#SBATCH -e logs/mac_on_hard_%A_%a.err

TASKS=(PR2 FC HN RD)
VICTIMS=(BiLSTM BERT)
ATTACKERS=(DeepWordBug BERTattack PWWS Genetic)

i=$SLURM_ARRAY_TASK_ID
ATK_IDX=$((i % 4))
VIC_IDX=$(((i / 4) % 2))
TASK_IDX=$((i / 8))

TASK=${TASKS[$TASK_IDX]}
VICTIM=${VICTIMS[$VIC_IDX]}
ATTACK=${ATTACKERS[$ATK_IDX]}

DATA_PATH="../BODEGA/data/$TASK"
MODEL_PATH="../BODEGA/data/$TASK/${VICTIM}-512.pth"
POLICY_PATH="models/${TASK}_${VICTIM}.pth"
OUT_DIR="results/online_true_hard"

CONDA_SH=/soft/easybuild/x86_64/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh
if [ ! -f "$CONDA_SH" ]; then
    echo "ERROR: conda.sh not found on $(hostname): $CONDA_SH" >&2
    exit 1
fi
source "$CONDA_SH"
conda activate bodega
export PYTHONPATH="../BODEGA:."
mkdir -p "$OUT_DIR" logs

echo "[$i] MACABEU ONLINE-TRUE (mv7/hard) | $TASK | $VICTIM | $ATTACK | warm=$POLICY_PATH"

python runs/eval_online.py \
    "$TASK" "$VICTIM" "$DATA_PATH" "$MODEL_PATH" "$OUT_DIR" \
    --attackers "$ATTACK" \
    --pretrained "$POLICY_PATH" \
    --label_source mv7 --reward_mode hard \
    --semantic_scorer BLEURT
