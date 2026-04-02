#!/bin/bash
# MACABEU: Cross-attacker generalization (leave-one-out)
# Usage: cd macabeu && bash scripts/run_cross_attacker.sh
#
# For each task and each held-out attacker:
#   1. Generate data with 3 of 4 attackers (excluding held-out)
#   2. Train agent on 3-attacker data (offline pre-training)
#   3. Evaluate OFFLINE (frozen policy) against ALL 4 attackers
#   4. Evaluate ONLINE (warm-started from offline) against ALL 4 attackers
#
# Two research questions:
#   - Offline: Do the 10 statistical features generalize to unseen attack types?
#   - Online:  Can online adaptation compensate for missing training data?

set -e

BODEGA_DIR="../BODEGA"
export PYTHONPATH="${BODEGA_DIR}:."

TASKS=(PR2 FC HN RD)
VICTIM="BiLSTM"
ALL_ATTACKERS=(DeepWordBug BERTattack PWWS Genetic)

for TASK in "${TASKS[@]}"; do
    DATA_PATH="${BODEGA_DIR}/data/${TASK}"
    MODEL_PATH="${DATA_PATH}/BiLSTM-512.pth"

    echo ""
    echo "=============================================="
    echo "  CROSS-ATTACKER: ${TASK} / ${VICTIM}"
    echo "=============================================="

    for HELD_OUT in "${ALL_ATTACKERS[@]}"; do
        echo ""
        echo "--- Held out: ${HELD_OUT} ---"

        # Build attacker list excluding held-out
        TRAIN_ATTACKERS=()
        for ATK in "${ALL_ATTACKERS[@]}"; do
            if [ "$ATK" != "$HELD_OUT" ]; then
                TRAIN_ATTACKERS+=("$ATK")
            fi
        done

        NPZ_FILE="agent_data/${TASK}_${VICTIM}_no_${HELD_OUT}.npz"
        POLICY_FILE="models/${TASK}_${VICTIM}_no_${HELD_OUT}.pth"
        OFFLINE_DIR="results/cross_attacker/${TASK}/no_${HELD_OUT}/offline"
        ONLINE_DIR="results/cross_attacker/${TASK}/no_${HELD_OUT}/online"

        # Step 1: Generate data with 3 attackers
        if [ -f "$NPZ_FILE" ]; then
            echo "[SKIP] Data exists: $NPZ_FILE"
        else
            echo "[1/4] Generating data (without ${HELD_OUT})..."
            python runs/generate_defense_data.py "$TASK" "$VICTIM" "$DATA_PATH" "$MODEL_PATH" agent_data/ \
                --attackers ${TRAIN_ATTACKERS[@]}
            # Rename to include held-out info
            mv "agent_data/${TASK}_${VICTIM}.npz" "$NPZ_FILE"
        fi

        # Step 2: Train offline agent
        if [ -f "$POLICY_FILE" ]; then
            echo "[SKIP] Model exists: $POLICY_FILE"
        else
            echo "[2/4] Training agent (without ${HELD_OUT})..."
            python runs/train_defense_agent.py "$NPZ_FILE" "$POLICY_FILE"
        fi

        # Step 3: Evaluate OFFLINE (frozen policy) against ALL 4 attackers
        LAST_OFFLINE="${OFFLINE_DIR}/results_${TASK}_False_Genetic_${VICTIM}_rl_defense.txt"
        if [ -f "$LAST_OFFLINE" ]; then
            echo "[SKIP] Offline results exist"
        else
            echo "[3/4] Evaluating offline (frozen) against all attackers..."
            mkdir -p "$OFFLINE_DIR"
            python runs/eval_defense_agent.py "$TASK" "$VICTIM" "$DATA_PATH" "$MODEL_PATH" "$POLICY_FILE" "$OFFLINE_DIR" \
                --semantic_scorer BLEURT
        fi

        # Step 4: Evaluate ONLINE (warm-started from offline) against ALL 4 attackers
        LAST_ONLINE="${ONLINE_DIR}/results_${TASK}_False_Genetic_${VICTIM}_online_rl.txt"
        if [ -f "$LAST_ONLINE" ]; then
            echo "[SKIP] Online results exist"
        else
            echo "[4/4] Evaluating online (warm start) against all attackers..."
            mkdir -p "$ONLINE_DIR"
            python runs/eval_online.py "$TASK" "$VICTIM" "$DATA_PATH" "$MODEL_PATH" "$ONLINE_DIR" \
                --pretrained "$POLICY_FILE" \
                --semantic_scorer BLEURT
        fi
    done
done

echo ""
echo "=============================================="
echo "  CROSS-ATTACKER EXPERIMENTS COMPLETE"
echo "=============================================="

# Print summary
echo ""
echo "=== Generalization Summary ==="
for TASK in "${TASKS[@]}"; do
    echo ""
    echo "--- ${TASK} ---"
    for HELD_OUT in "${ALL_ATTACKERS[@]}"; do
        echo "  Held out: ${HELD_OUT}"
        for ATK in "${ALL_ATTACKERS[@]}"; do
            MARKER=""
            if [ "$ATK" = "$HELD_OUT" ]; then
                MARKER=" <-- UNSEEN"
            fi

            OFFLINE_RESULT="results/cross_attacker/${TASK}/no_${HELD_OUT}/offline/results_${TASK}_False_${ATK}_${VICTIM}_rl_defense.txt"
            ONLINE_RESULT="results/cross_attacker/${TASK}/no_${HELD_OUT}/online/results_${TASK}_False_${ATK}_${VICTIM}_online_rl.txt"

            OFFLINE_SCORE="-"
            ONLINE_SCORE="-"
            if [ -f "$OFFLINE_RESULT" ]; then
                OFFLINE_SCORE=$(grep "BODEGA score:" "$OFFLINE_RESULT" | head -1 | awk '{print $NF}')
            fi
            if [ -f "$ONLINE_RESULT" ]; then
                ONLINE_SCORE=$(grep "BODEGA score:" "$ONLINE_RESULT" | head -1 | awk '{print $NF}')
            fi

            echo "    vs ${ATK}: offline=${OFFLINE_SCORE}  online=${ONLINE_SCORE}${MARKER}"
        done
    done
done
