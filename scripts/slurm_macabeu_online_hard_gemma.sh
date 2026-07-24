#!/bin/bash
# SLURM array: standard attackers vs TRUE-ONLINE MACABEU (mv7/hard) on GEMMA.
# Fills the GEMMA hole in the hard grid so Table 1's MACABEU-est row covers
# the full 12 sub-combos. 16 cells = 4 tasks x 4 attackers x GEMMA.
#
# Submit:
#   sbatch --exclude=node023 scripts/slurm_macabeu_online_hard_gemma.sh

#SBATCH -J mac_on_hard_g
#SBATCH -p high
#SBATCH --gres=gpu:1
#SBATCH --constraint=cuda
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH --array=0-15
#SBATCH -o logs/mac_on_hard_g_%A_%a.out
#SBATCH -e logs/mac_on_hard_g_%A_%a.err

TASKS=(PR2 FC HN RD)
ATTACKERS=(DeepWordBug BERTattack PWWS Genetic)

i=$SLURM_ARRAY_TASK_ID
ATK_IDX=$((i % 4))
TASK_IDX=$((i / 4))

TASK=${TASKS[$TASK_IDX]}
VICTIM=GEMMA
ATTACK=${ATTACKERS[$ATK_IDX]}

DATA_PATH="../BODEGA/data/$TASK"
MODEL_PATH="../BODEGA/data/$TASK/GEMMA-512"
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

echo "[$i] MACABEU ONLINE-TRUE (mv7/hard) GEMMA-fill | $TASK | $VICTIM | $ATTACK | warm=$POLICY_PATH"

python runs/eval_online.py \
    "$TASK" "$VICTIM" "$DATA_PATH" "$MODEL_PATH" "$OUT_DIR" \
    --attackers "$ATTACK" \
    --pretrained "$POLICY_PATH" \
    --label_source mv7 --reward_mode hard \
    --semantic_scorer BLEURT
