#!/bin/bash
# SLURM array: clean accuracy for MACABEU-Oracle and MACABEU-est (mv7/hard) on
# the clean dev split, with online adaptation enabled (matches deployment).
# 24 cells = 4 tasks x 3 victims x 2 label sources.
#
# Submit:
#   sbatch scripts/slurm_macabeu_clean_accuracy_online.sh

#SBATCH -J mac_clean_on
#SBATCH -p high
#SBATCH --gres=gpu:1
#SBATCH --constraint=cuda
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH --array=0-23
#SBATCH -o logs/mac_clean_on_%A_%a.out
#SBATCH -e logs/mac_clean_on_%A_%a.err

TASKS=(PR2 FC HN RD)
VICTIMS=(BiLSTM BERT GEMMA)
LABEL_SOURCES=(oracle mv7)

i=$SLURM_ARRAY_TASK_ID
# Layout: task-major, victim-mid, label-source-inner
LS_IDX=$((i % 2))
VIC_IDX=$(((i / 2) % 3))
TASK_IDX=$((i / 6))

TASK=${TASKS[$TASK_IDX]}
VICTIM=${VICTIMS[$VIC_IDX]}
LSRC=${LABEL_SOURCES[$LS_IDX]}
REWARD=hard   # per Piotr's ask we ship the discrete/hard variant

DATA_PATH="../BODEGA/data/$TASK"
case "$VICTIM" in
    GEMMA) MODEL_PATH="../BODEGA/data/$TASK/GEMMA-512" ;;
    *)     MODEL_PATH="../BODEGA/data/$TASK/${VICTIM}-512.pth" ;;
esac
POLICY_PATH="models/${TASK}_${VICTIM}.pth"
OUT_DIR="results/clean_accuracy_online"

CONDA_SH=/soft/easybuild/x86_64/software/Anaconda3/2023.09-0/etc/profile.d/conda.sh
if [ ! -f "$CONDA_SH" ]; then
    echo "ERROR: conda.sh not found on $(hostname): $CONDA_SH" >&2
    exit 1
fi
source "$CONDA_SH"
conda activate bodega
export PYTHONPATH="../BODEGA:."
mkdir -p "$OUT_DIR" logs

echo "[$i] MACABEU-clean ($LSRC/$REWARD) | $TASK | $VICTIM | warm=$POLICY_PATH"

python runs/eval_clean_accuracy_online.py \
    "$TASK" "$VICTIM" "$DATA_PATH" "$MODEL_PATH" "$POLICY_PATH" "$OUT_DIR" \
    --label_source "$LSRC" --reward_mode "$REWARD"
