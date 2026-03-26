#!/bin/bash
# MACABEU: End-to-end RL defense selector experiment
# Usage: cd macabeu && bash scripts/run_experiment.sh
set -e

BODEGA_PATH="../BODEGA"
export PYTHONPATH="$BODEGA_PATH:.:$PYTHONPATH"

TASK="${1:-PR2}"
VICTIM="${2:-BiLSTM}"
DATA_PATH="$BODEGA_PATH/data/$TASK"
MODEL_PATH="$BODEGA_PATH/data/$TASK/${VICTIM}-512.pth"

echo "=== MACABEU: RL Defense Selector ==="
echo "Task: $TASK, Victim: $VICTIM"
echo ""

# Step 1: Generate training data
echo "=== Step 1: Generating training data ==="
python runs/generate_defense_data.py \
    "$TASK" "$VICTIM" "$DATA_PATH" "$MODEL_PATH" agent_data/

echo ""

# Step 2: Train agent
echo "=== Step 2: Training RL agent ==="
python runs/train_defense_agent.py \
    "agent_data/${TASK}_${VICTIM}.npz" \
    "models/${TASK}_${VICTIM}.pth"

echo ""

# Step 3: Evaluate against all attackers
echo "=== Step 3: Evaluating RL defense selector ==="
python runs/eval_defense_agent.py \
    "$TASK" "$VICTIM" "$DATA_PATH" "$MODEL_PATH" \
    "models/${TASK}_${VICTIM}.pth" \
    results/

echo ""
echo "=== Done! Results in results/ ==="
