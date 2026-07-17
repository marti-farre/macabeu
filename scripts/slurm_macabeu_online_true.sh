#!/bin/bash
# SLURM array: standard attackers vs TRUE-ONLINE MACABEU (mv7/soft).
# 48 cells = 4 tasks × 3 victims × 4 attackers. One job per cell so cheap
# BiLSTM cells finish fast while slow GEMMA cells run in parallel.
#
# Submit:
#   sbatch scripts/slurm_macabeu_online_true.sh

#SBATCH -J mac_on_true
#SBATCH -p high
#SBATCH --gres=gpu:1
#SBATCH --constraint=cuda
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH --array=0-47
#SBATCH -o logs/mac_on_true_%A_%a.out
#SBATCH -e logs/mac_on_true_%A_%a.err

TASKS=(PR2 FC HN RD)
VICTIMS=(BiLSTM BERT GEMMA)
ATTACKERS=(DeepWordBug BERTattack PWWS Genetic)

i=$SLURM_ARRAY_TASK_ID
# Index: TASK (0-3), VICTIM (0-2), ATTACKER (0-3); layout task-major, victim-mid, attacker-inner
ATK_IDX=$((i % 4))
VIC_IDX=$(((i / 4) % 3))
TASK_IDX=$((i / 12))

TASK=${TASKS[$TASK_IDX]}
VICTIM=${VICTIMS[$VIC_IDX]}
ATTACK=${ATTACKERS[$ATK_IDX]}

DATA_PATH="../BODEGA/data/$TASK"
case "$VICTIM" in
    GEMMA) MODEL_PATH="../BODEGA/data/$TASK/GEMMA-512" ;;
    *)     MODEL_PATH="../BODEGA/data/$TASK/${VICTIM}-512.pth" ;;
esac
POLICY_PATH="models/${TASK}_${VICTIM}.pth"
OUT_DIR="results/online_true_soft"

CONDA_SH=/soft/easybuild/x86_64/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh
if [ ! -f "$CONDA_SH" ]; then
    echo "ERROR: conda.sh not found on $(hostname): $CONDA_SH" >&2
    exit 1
fi
source "$CONDA_SH"
conda activate bodega
export PYTHONPATH="../BODEGA:."
mkdir -p "$OUT_DIR" logs

echo "[$i] MACABEU ONLINE-TRUE (mv7/soft) | $TASK | $VICTIM | $ATTACK | warm=$POLICY_PATH"

python runs/eval_online.py \
    "$TASK" "$VICTIM" "$DATA_PATH" "$MODEL_PATH" "$OUT_DIR" \
    --attackers "$ATTACK" \
    --pretrained "$POLICY_PATH" \
    --label_source mv7 --reward_mode soft \
    --semantic_scorer BLEURT
