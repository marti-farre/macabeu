#!/bin/bash
# SLURM job: fill-in MACABEU eval steps for a single Gemma task.
# Runs ONLY the missing offline attackers + all 4 online attackers.
#
# Usage:
#   cd ~/macabeu
#   # RD: Genetic offline missing, all 4 online missing
#   sbatch --export=ALL,MAC_TASK=RD,MAC_OFFLINE='Genetic',MAC_ONLINE='DeepWordBug BERTattack PWWS Genetic' \
#       scripts/slurm_macabeu_fill_gemma.sh
#   # HN (after mac_gemma HN finishes — check which are missing first):
#   sbatch --export=ALL,MAC_TASK=HN,MAC_OFFLINE='BERTattack PWWS Genetic',MAC_ONLINE='DeepWordBug BERTattack PWWS Genetic' \
#       scripts/slurm_macabeu_fill_gemma.sh

#SBATCH -J mac_fill
#SBATCH -p high
#SBATCH --gres=gpu:1
#SBATCH --constraint=cuda
#SBATCH --mem=48G
#SBATCH -c 4
#SBATCH -o logs/mac_fill_%j.out
#SBATCH -e logs/mac_fill_%j.err

TASK="${MAC_TASK:?must set MAC_TASK (e.g. RD, HN)}"
VICTIM="GEMMA"
OFFLINE_ATTACKERS="${MAC_OFFLINE:-}"
ONLINE_ATTACKERS="${MAC_ONLINE:-DeepWordBug BERTattack PWWS Genetic}"

DATA_PATH="../BODEGA/data/$TASK"
MODEL_PATH="../BODEGA/data/$TASK/${VICTIM}-512"
POLICY_FILE="models/${TASK}_${VICTIM}.pth"

source /soft/easybuild/x86_64/software/Miniconda3/22.11.1-1/etc/profile.d/conda.sh
conda activate bodega
export PYTHONPATH="../BODEGA:."
mkdir -p results/offline results/online logs

echo "=== MACABEU FILL | $TASK | $VICTIM ==="
echo "Offline attackers: '$OFFLINE_ATTACKERS'"
echo "Online attackers:  '$ONLINE_ATTACKERS'"

if [ ! -f "$POLICY_FILE" ]; then
    echo "ERROR: policy not found: $POLICY_FILE"
    exit 1
fi

# Step 3 (offline) — only requested attackers
if [ -n "$OFFLINE_ATTACKERS" ]; then
    echo "[Step 1/2] Offline eval for: $OFFLINE_ATTACKERS"
    python runs/eval_defense_agent.py \
        "$TASK" "$VICTIM" "$DATA_PATH" "$MODEL_PATH" "$POLICY_FILE" results/offline/ \
        --attackers $OFFLINE_ATTACKERS \
        --semantic_scorer BLEURT
else
    echo "[Step 1/2] No offline attackers requested — skipping"
fi

# Step 4 (online) — only requested attackers
if [ -n "$ONLINE_ATTACKERS" ]; then
    echo "[Step 2/2] Online eval for: $ONLINE_ATTACKERS"
    python runs/eval_online.py \
        "$TASK" "$VICTIM" "$DATA_PATH" "$MODEL_PATH" results/online/ \
        --pretrained "$POLICY_FILE" \
        --attackers $ONLINE_ATTACKERS \
        --semantic_scorer BLEURT
else
    echo "[Step 2/2] No online attackers requested — skipping"
fi

echo "=== DONE: $TASK $VICTIM ==="
