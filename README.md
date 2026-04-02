# MACABEU: RL-Based Adaptive Defense Selector for BODEGA

MACABEU is a reinforcement learning agent that dynamically selects which preprocessing defense to apply to each input text at inference time, protecting NLP classifiers against adversarial attacks. It uses a **contextual bandit** formulation: a lightweight Q-network maps 10 statistical text features to one of 8 defense actions, choosing the best defense per input.

MACABEU is the defense counterpart to [XARELLO](https://aclanthology.org/2024.wassa-1.11/) (RL for adaptive attack). While XARELLO learns *which parts of the sentence can be modified to succesfully attack* , MACABEU learns *which defense* to apply.

Built on top of the [BODEGA](https://doi.org/10.1017/nlp.2024.54) benchmark for adversarial robustness in misinformation detection.

## Architecture

```
                          MACABEU Pipeline
 ┌─────────┐    ┌──────────────────┐    ┌───────────┐    ┌────────┐
 │  Input   │───>│ Feature Extractor │───>│ Q-Network │───>│ Select │
 │  text    │    │  (10 statistics)  │    │   (MLP)   │    │ action │
 └─────────┘    └──────────────────┘    └───────────┘    └───┬────┘
                                                             │
                    ┌────────────────────────────────────────┘
                    v
 ┌──────────────────────────────────────────────────────────────────┐
 │  Action Space (8 defenses)                                       │
 │  0: none          3: majority_vote@3   6: spellcheck_mv@3       │
 │  1: spellcheck    4: majority_vote@7   7: char_noise@0.10       │
 │  2: unicode       5: discretize                                  │
 └──────────────────────────────┬───────────────────────────────────┘
                                v
                         ┌──────────┐    ┌────────────┐
                         │  Apply   │───>│   Victim   │───> prediction
                         │ defense  │    │  classifier│
                         └──────────┘    └────────────┘
```

**State** (10 features): `text_length`, `word_count`, `avg_word_length`, `oov_ratio`, `non_ascii_ratio`, `uppercase_ratio`, `punctuation_ratio`, `digit_ratio`, `repeated_char_ratio`, `char_entropy`

**Q-Network**: `Linear(10,64) → ReLU → Linear(64,32) → ReLU → Linear(32,8)` (~3K parameters)

**Reward**: +1.0 if correct after defense, -1.0 if wrong, minus small cost for expensive defenses.

## Installation

### Prerequisites

1. Clone BODEGA as a sibling directory:
   ```
   git clone <bodega-repo-url> BODEGA
   git clone <macabeu-repo-url> macabeu
   ```

2. Set up the conda environment (shared with BODEGA):
   ```bash
   conda create -n bodega python=3.10
   conda activate bodega
   conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
   pip install "transformers==4.38.1"
   pip install OpenAttack
   pip install bert-score editdistance
   pip install symspellpy    # for OOV feature extraction
   ```

3. Set PYTHONPATH:
   ```bash
   export PYTHONPATH="../BODEGA:."
   ```

### Data

MACABEU uses BODEGA's data and trained victim models. Ensure you have:
- `../BODEGA/data/{TASK}/train.tsv` and `attack.tsv` for each task
- `../BODEGA/data/{TASK}/BiLSTM-512.pth` (trained victim model)

See the BODEGA README for data preparation instructions.

## Quick Start

Run the full pipeline for a single task:

```bash
cd macabeu
export PYTHONPATH="../BODEGA:."
bash scripts/run_experiment.sh PR2 BiLSTM
```

This runs three steps:

1. **Generate training data** — runs 4 attackers (DeepWordBug, BERTattack, PWWS, Genetic) on the attack set, evaluates all 8 defenses per example, saves features + rewards to `agent_data/PR2_BiLSTM.npz`
2. **Train Q-network** — offline RL on the generated data (~1 minute), saves model to `models/PR2_BiLSTM.pth`
3. **Evaluate** — runs each attacker against the RL defense selector, reports BODEGA scores to `results/`

Or run each step individually:

```bash
# Step 1: Generate data
python runs/generate_defense_data.py PR2 BiLSTM ../BODEGA/data/PR2 ../BODEGA/data/PR2/BiLSTM-512.pth agent_data/

# Step 2: Train
python runs/train_defense_agent.py agent_data/PR2_BiLSTM.npz models/PR2_BiLSTM.pth

# Step 3a: Evaluate offline (frozen policy)
python runs/eval_defense_agent.py PR2 BiLSTM ../BODEGA/data/PR2 ../BODEGA/data/PR2/BiLSTM-512.pth \
    models/PR2_BiLSTM.pth results/offline/PR2/

# Step 3b: Evaluate online (learns during attack)
python runs/eval_online.py PR2 BiLSTM ../BODEGA/data/PR2 ../BODEGA/data/PR2/BiLSTM-512.pth \
    results/online/PR2/
```

## Experiments

### Multi-task evaluation

Run the full pipeline for all 4 BODEGA tasks (PR2, FC, HN, RD) with both offline and online evaluation:

```bash
bash scripts/run_all_tasks.sh
```

This produces 32 experiments: 4 tasks × 4 attackers × 2 modes (offline + online). Skips steps where output files already exist.

### Cross-attacker generalization

Test whether the agent can defend against unseen attack types using leave-one-out:

```bash
bash scripts/run_cross_attacker.sh
```

For each task, trains on 3 of 4 attackers and evaluates against all 4 in both modes:
- **Offline (frozen)**: Do the statistical features generalize to unseen attacks?
- **Online (warm start)**: Can online adaptation compensate for missing training data?

Total: 128 experiments (4 tasks × 4 held-out × 4 eval attackers × 2 modes).

## Evaluation Modes

| Mode | Description | Use case |
|------|-------------|----------|
| **Offline** | Frozen pre-trained policy. No learning at inference time. | Baseline: test pure generalization of learned policy |
| **Online** | Learns during the attack via epsilon-greedy exploration + replay buffer updates. | Realistic deployment: agent adapts to new attacks in real-time |

Online mode can optionally warm-start from an offline model (`--pretrained` flag in `eval_online.py`).

## Action Space

| Index | Defense | Description | Cost |
|-------|---------|-------------|------|
| 0 | `none` | No defense (baseline) | 0.00 |
| 1 | `spellcheck` | SymSpell-based spelling correction | 0.00 |
| 2 | `unicode` | Unicode homoglyph normalization | 0.00 |
| 3 | `majority_vote@3` | Run 3 noisy copies, majority vote | 0.05 |
| 4 | `majority_vote@7` | Run 7 noisy copies, majority vote | 0.10 |
| 5 | `discretize` | Quantize embeddings | 0.00 |
| 6 | `spellcheck_mv@3` | Spellcheck + 3x majority vote | 0.05 |
| 7 | `char_noise@0.10` | Add 10% Unicode noise | 0.00 |

## Project Structure

```
macabeu/
├── agent/                        # RL agent implementation
│   ├── features.py               #   10 statistical text features
│   ├── q_network.py              #   Q-network MLP + epsilon-greedy policy
│   ├── replay_buffer.py          #   Experience replay buffer
│   ├── defense_env.py            #   Defense environment (evaluates all 8 defenses)
│   ├── defense_selector.py       #   Offline selector (frozen policy)
│   └── online_selector.py        #   Online selector (learns during attack)
├── runs/                         # Execution scripts
│   ├── generate_defense_data.py  #   Generate NPZ training data with real attackers
│   ├── train_defense_agent.py    #   Train Q-network offline
│   ├── eval_defense_agent.py     #   Evaluate offline policy
│   └── eval_online.py            #   Evaluate online learning
├── scripts/                      # Shell automation
│   ├── run_experiment.sh         #   Single task pipeline
│   ├── run_all_tasks.sh          #   All 4 tasks, offline + online
│   └── run_cross_attacker.sh     #   Leave-one-out generalization
├── explanations/                 # Experiment documentation
├── agent_data/                   #   Generated NPZ training data (gitignored)
├── models/                       #   Trained policy checkpoints (gitignored)
└── results/                      #   Experiment outputs (gitignored)
```

## References

- **BODEGA**: [Verifying the Robustness of Automatic Credibility Assessment](https://doi.org/10.1017/nlp.2024.54) (NLP Journal)
- **XARELLO**: [Know Thine Enemy: Adaptive Attacks on Misinformation Detection Using Reinforcement Learning](https://aclanthology.org/2024.wassa-1.11/) (WASSA @ ACL 2024)
- Developed within the [ERINIA](https://www.upf.edu/web/erinia) project at the [TALN lab](https://www.upf.edu/web/taln/), Universitat Pompeu Fabra.
