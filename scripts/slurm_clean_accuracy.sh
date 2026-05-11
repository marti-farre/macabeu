#!/bin/bash
# SLURM array job: Clean accuracy evaluation for MACABEU RL defense
# Covers all 12 (task, victim) combinations:
#   tasks   = PR2, FC, HN, RD
#   victims = BiLSTM, BERT, GEMMA
# For each combo: baseline (no defense), every static defense in the action
# space, and the trained MACABEU offline policy.
#
# Submit with: cd ~/macabeu && sbatch scripts/slurm_clean_accuracy.sh

#SBATCH -J mac_acc
#SBATCH -p high
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -c 4
#SBATCH --array=0-11
#SBATCH -o logs/mac_clean_acc_%A_%a.out
#SBATCH -e logs/mac_clean_acc_%A_%a.err

TASKS=(PR2 FC HN RD)
VICTIMS=(BiLSTM BERT GEMMA)

i=$SLURM_ARRAY_TASK_ID
TASK_IDX=$((i / 3))
VIC_IDX=$((i % 3))
TASK=${TASKS[$TASK_IDX]}
VICTIM=${VICTIMS[$VIC_IDX]}

DATA_PATH="../BODEGA/data/$TASK"
if [ "$VICTIM" = "GEMMA" ]; then
    MODEL_PATH="../BODEGA/data/$TASK/${VICTIM}-512"
else
    MODEL_PATH="../BODEGA/data/$TASK/${VICTIM}-512.pth"
fi
POLICY_PATH="models/${TASK}_${VICTIM}.pth"
OUT_DIR="results/clean_accuracy"

module load Miniconda3
eval "$(conda shell.bash hook)"
conda activate bodega
export PYTHONPATH="../BODEGA:."
mkdir -p "$OUT_DIR" logs

echo "[$i] MACABEU Clean Accuracy | $TASK | $VICTIM"
echo "    data    = $DATA_PATH"
echo "    model   = $MODEL_PATH"
echo "    policy  = $POLICY_PATH"

if [ ! -f "$POLICY_PATH" ]; then
    echo "WARN: policy $POLICY_PATH missing; skipping MACABEU comparison."
fi

python runs/eval_clean_accuracy.py \
    "$TASK" "$VICTIM" "$DATA_PATH" "$MODEL_PATH" "$POLICY_PATH" "$OUT_DIR" \
    --static_defenses
