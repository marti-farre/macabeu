# Branch: experiment/multi-task-bilstm

## Overview

**Purpose**: Validate that the MACABEU RL defense selector generalizes across NLP tasks and attack types beyond the initial PR2/BiLSTM setup.

**Parent Branch**: `main`

**Status**: In progress

---

## What is MACABEU?

MACABEU is an RL-based **adaptive defense selector** that dynamically chooses which preprocessing defense to apply to each input text during inference. It mirrors XARELLO (RL for adaptive attack) but flips the paradigm: RL for defense.

### Architecture

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
 └──────────────────────────────────┬───────────────────────────────┘
                                    v
                             ┌──────────┐    ┌────────────┐
                             │  Apply   │───>│   Victim   │───> prediction
                             │ defense  │    │  classifier│
                             └──────────┘    └────────────┘
```

### State: 10 Statistical Text Features
- `text_length`, `word_count`, `avg_word_length`
- `oov_ratio` (SymSpell dictionary: captures misspellings from char-level attacks)
- `non_ascii_ratio` (homoglyph signature)
- `uppercase_ratio`, `punctuation_ratio`, `digit_ratio`
- `repeated_char_ratio` (DeepWordBug signature)
- `char_entropy`

### Formulation
- **Contextual bandit** (1-step MDP, gamma=0): choose defense -> observe result
- **Reward**: +1.0 if correct after defense, -1.0 if wrong; small cost for expensive defenses
- **Q-Network**: `Linear(10,64) -> ReLU -> Linear(64,32) -> ReLU -> Linear(32,8)`
- **Two modes**: Offline (frozen policy from pre-trained data) and Online (learns during attack)

---

## MACABEU vs XARELLO Comparison

| Aspect | XARELLO (RL Attack) | MACABEU (RL Defense) |
|--------|---------------------|----------------------|
| **Goal** | Flip classifier prediction | Protect classifier from attacks |
| **Formulation** | Full MDP (gamma=0.5) | Contextual bandit (gamma=0) |
| **State** | BERT embeddings (768d) + candidates | 10 statistical text features |
| **Actions** | Token position x replacement (~10K) | 8 defense choices |
| **Steps/episode** | 5 steps x 5 episodes = 25 modifications | 1 step (single defense choice) |
| **Training** | 1,600 texts x 25 steps = 40K transitions | ~2,000 examples x 1 step |
| **Training time** | Hours (800K+ victim queries) | ~1 minute (~1K updates) |
| **Network** | BERT backbone + linear (~110M params) | 3-layer MLP (~3K params) |

---

## Experiment Matrix

### Multi-Task (this branch)

| | BiLSTM (offline) | BiLSTM (online) |
|---|---|---|
| **PR2** | done | done |
| **FC** | running | running |
| **HN** | running | running |
| **RD** | running | running |

- **Attackers**: DeepWordBug, BERTattack, PWWS, Genetic
- **Metric**: BODEGA score (BERTscore semantic similarity)
- **Total**: 4 tasks x 4 attackers x 2 modes = 32 experiments

### Cross-Attacker Generalization

For each task, train on 3 of 4 attackers, evaluate against all 4 with **both** modes:
- Train without DeepWordBug, test on DeepWordBug (can it generalize to char-level?)
- Train without BERTattack, test on BERTattack
- Train without PWWS, test on PWWS
- Train without Genetic, test on Genetic

Two evaluation modes per configuration:
- **Offline (frozen)**: Pure generalization — do the features transfer to unseen attacks?
- **Online (warm start)**: Adaptation — can online learning compensate for missing training?

**Total**: 4 tasks x 4 held-out x 4 eval attackers x 2 modes = 128 experiments

---

## Existing Results: PR2/BiLSTM

### Offline RL (frozen policy)

| Attack | Success | Semantic | Char | BODEGA |
|--------|---------|----------|------|--------|
| DeepWordBug | 0.1707 | 0.4935 | 0.9439 | 0.0810 |
| BERTattack | 0.3197 | 0.8116 | 0.9430 | 0.2465 |
| PWWS | 0.3413 | 0.7309 | 0.8913 | 0.2282 |
| Genetic | 0.3846 | 0.7667 | 0.9158 | 0.2742 |
| **Average** | | | | **0.2074** |

### Online RL (learns during attack)

| Attack | Success | Semantic | Char | BODEGA |
|--------|---------|----------|------|--------|
| DeepWordBug | 0.1301 | 0.5318 | 0.9416 | 0.0664 |
| BERTattack | 0.2738 | 0.7639 | 0.9197 | 0.1938 |
| PWWS | 0.2481 | 0.6780 | 0.8368 | 0.1451 |
| Genetic | 0.2000 | 0.7486 | 0.9051 | 0.1382 |
| **Average** | | | | **0.1359** |

### Comparison with No Defense

| Attack | No Defense BODEGA | Online RL BODEGA | Reduction |
|--------|------------------|-----------------|-----------|
| DeepWordBug | 0.248 | 0.066 | -73% |
| BERTattack | 0.546 | 0.194 | -64% |
| PWWS | 0.587 | 0.145 | -75% |
| Genetic | 0.595 | 0.138 | -77% |
| **Average** | **0.494** | **0.136** | **-72%** |

---

## Files

| File | Description |
|------|-------------|
| `agent/features.py` | TextFeatureExtractor (10 statistical features) |
| `agent/q_network.py` | DefenseQNetwork MLP + DefensePolicy (epsilon-greedy) |
| `agent/replay_buffer.py` | Experience replay buffer |
| `agent/defense_env.py` | DefenseEnvironment (evaluates all 8 defenses) |
| `agent/defense_selector.py` | RLDefenseSelector (offline, frozen policy) |
| `agent/online_selector.py` | OnlineRLDefenseSelector (learns during attack) |
| `runs/generate_defense_data.py` | Generate NPZ training data with real attackers |
| `runs/train_defense_agent.py` | Train Q-network offline |
| `runs/eval_defense_agent.py` | Evaluate offline policy |
| `runs/eval_online.py` | Evaluate online learning |
| `scripts/run_all_tasks.sh` | **New** - Full pipeline for all 4 tasks |
| `scripts/run_cross_attacker.sh` | **New** - Leave-one-out attacker generalization |
| `explanations/experiment_multi-task-bilstm.md` | **New** - This file |

---

## Usage

```bash
# Full pipeline for all tasks
cd macabeu
bash scripts/run_all_tasks.sh

# Single task
PYTHONPATH=../BODEGA:. bash scripts/run_experiment.sh PR2 BiLSTM

# Cross-attacker generalization
bash scripts/run_cross_attacker.sh
```

---

## Result Files

```
results/
├── offline/
│   ├── PR2/
│   │   ├── results_PR2_False_{attack}_BiLSTM_rl_defense.txt
│   │   └── summary_PR2_BiLSTM_rl_defense.txt
│   ├── FC/ ...
│   ├── HN/ ...
│   └── RD/ ...
├── online/
│   ├── PR2/
│   │   ├── results_PR2_False_{attack}_BiLSTM_online_rl.txt
│   │   └── online_model_PR2_BiLSTM_{attack}.pth
│   ├── FC/ ...
│   ├── HN/ ...
│   └── RD/ ...
└── cross_attacker/
    ├── PR2/
    │   ├── no_{held_out}/
    │   │   ├── offline/   (frozen policy results)
    │   │   │   ├── results_PR2_False_{attack}_BiLSTM_rl_defense.txt
    │   │   │   └── summary_PR2_BiLSTM_rl_defense.txt
    │   │   └── online/    (warm-started online results)
    │   │       └── results_PR2_False_{attack}_BiLSTM_online_rl.txt
    │   └── ...
    ├── FC/ ...
    ├── HN/ ...
    └── RD/ ...
```

---

## Next Steps

- [ ] Collect results for all 4 tasks
- [ ] Cross-attacker generalization analysis
- [ ] Compare against best static defense per task (from BODEGA experiment-7)
- [ ] Cross-victim: train on BiLSTM, test on BERT/Gemma (needs HPC)
- [ ] Cross-task: train on one task, test on others
