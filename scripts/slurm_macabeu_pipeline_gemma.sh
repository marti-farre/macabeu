#!/bin/bash
# SLURM array job: Full MACABEU pipeline for Gemma
# generate data → train → offline eval → online eval
# 4 jobs (one per task)
#
# Submit with: cd ~/macabeu && sbatch scripts/slurm_macabeu_pipeline_gemma.sh

#SBATCH -J mac_gemma
#SBATCH -p high
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -c 4
#SBATCH --array=0-3
#SBATCH -o logs/macabeu_gemma_%A_%a.out
#SBATCH -e logs/macabeu_gemma_%A_%a.err

TASKS=(PR2 FC HN RD)

i=$SLURM_ARRAY_TASK_ID
TASK=${TASKS[$i]}

VICTIM="GEMMA"
DATA_PATH="../BODEGA/data/$TASK"
MODEL_PATH="../BODEGA/data/$TASK/GEMMA-512"

module load Miniconda3
eval "$(conda shell.bash hook)"
conda activate bodega
export PYTHONPATH="../BODEGA:."
mkdir -p agent_data models results/offline results/online logs

echo "=== [$i] MACABEU FULL PIPELINE | $TASK | $VICTIM ==="

# Step 1: Generate defense data
NPZ_FILE="agent_data/${TASK}_${VICTIM}.npz"
if [ ! -f "$NPZ_FILE" ]; then
    echo "[Step 1/4] Generating defense data..."
    python runs/generate_defense_data.py \
        "$TASK" "$VICTIM" "$DATA_PATH" "$MODEL_PATH" agent_data/
else
    echo "[Step 1/4] Data already exists: $NPZ_FILE — skipping"
fi

# Step 2: Train RL agent
POLICY_FILE="models/${TASK}_${VICTIM}.pth"
if [ ! -f "$POLICY_FILE" ]; then
    echo "[Step 2/4] Training RL agent..."
    python runs/train_defense_agent.py \
        "$NPZ_FILE" "$POLICY_FILE"
else
    echo "[Step 2/4] Policy already exists: $POLICY_FILE — skipping"
fi

# Step 3: Offline evaluation
echo "[Step 3/4] Offline evaluation..."
python runs/eval_defense_agent.py \
    "$TASK" "$VICTIM" "$DATA_PATH" "$MODEL_PATH" "$POLICY_FILE" results/offline/ \
    --semantic_scorer BLEURT

# Step 4: Online evaluation (warm-started from offline policy)
echo "[Step 4/4] Online evaluation..."
python runs/eval_online.py \
    "$TASK" "$VICTIM" "$DATA_PATH" "$MODEL_PATH" results/online/ \
    --pretrained "$POLICY_FILE" \
    --semantic_scorer BLEURT

echo "=== DONE: $TASK $VICTIM ==="
