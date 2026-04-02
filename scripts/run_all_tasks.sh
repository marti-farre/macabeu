#!/bin/bash
# MACABEU: Run full pipeline for all 4 tasks (BiLSTM victim, BLEURT)
# Usage: cd macabeu && bash scripts/run_all_tasks.sh
#
# Pipeline per task:
#   1. Generate defense data (adversarial examples + defense rewards)
#   2. Train RL agent
#   3. Evaluate offline policy vs all attackers
#   4. Evaluate online learning vs all attackers
#
# Skips steps where output files already exist.

set -e

BODEGA_DIR="../BODEGA"
export PYTHONPATH="${BODEGA_DIR}:."

TASKS=(PR2 FC HN RD)
VICTIM="BiLSTM"

for TASK in "${TASKS[@]}"; do
    DATA_PATH="${BODEGA_DIR}/data/${TASK}"
    MODEL_PATH="${DATA_PATH}/BiLSTM-512.pth"
    NPZ_FILE="agent_data/${TASK}_${VICTIM}.npz"
    POLICY_FILE="models/${TASK}_${VICTIM}.pth"

    echo ""
    echo "=============================================="
    echo "  TASK: ${TASK} / ${VICTIM}"
    echo "=============================================="

    # Step 1: Generate data (skip if NPZ already exists)
    if [ -f "$NPZ_FILE" ]; then
        echo "[SKIP] Data already exists: $NPZ_FILE"
    else
        echo "[1/4] Generating defense data..."
        python runs/generate_defense_data.py "$TASK" "$VICTIM" "$DATA_PATH" "$MODEL_PATH" agent_data/
    fi

    # Step 2: Train agent (skip if model already exists)
    if [ -f "$POLICY_FILE" ]; then
        echo "[SKIP] Model already exists: $POLICY_FILE"
    else
        echo "[2/4] Training RL agent..."
        python runs/train_defense_agent.py "$NPZ_FILE" "$POLICY_FILE"
    fi

    # Step 3: Evaluate offline
    OFFLINE_DIR="results/offline/${TASK}"
    if [ -f "${OFFLINE_DIR}/summary_${TASK}_${VICTIM}_rl_defense.txt" ]; then
        echo "[SKIP] Offline results exist"
    else
        echo "[3/4] Evaluating offline policy..."
        mkdir -p "$OFFLINE_DIR"
        python runs/eval_defense_agent.py "$TASK" "$VICTIM" "$DATA_PATH" "$MODEL_PATH" "$POLICY_FILE" "$OFFLINE_DIR" \
            --semantic_scorer BLEURT
    fi

    # Step 4: Evaluate online
    ONLINE_DIR="results/online/${TASK}"
    if [ -f "${ONLINE_DIR}/results_${TASK}_False_Genetic_${VICTIM}_online_rl.txt" ]; then
        echo "[SKIP] Online results exist"
    else
        echo "[4/4] Evaluating online learning..."
        mkdir -p "$ONLINE_DIR"
        python runs/eval_online.py "$TASK" "$VICTIM" "$DATA_PATH" "$MODEL_PATH" "$ONLINE_DIR" \
            --semantic_scorer BLEURT
    fi
done

echo ""
echo "=============================================="
echo "  ALL TASKS COMPLETE"
echo "=============================================="
echo ""

# Print summary across tasks
for TASK in "${TASKS[@]}"; do
    OFFLINE_SUMMARY="results/offline/${TASK}/summary_${TASK}_${VICTIM}_rl_defense.txt"
    if [ -f "$OFFLINE_SUMMARY" ]; then
        echo "--- ${TASK} (offline) ---"
        cat "$OFFLINE_SUMMARY"
        echo ""
    fi
done
