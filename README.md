# MACABEU: RL-Based Adaptive Defense Selector for BODEGA

> **Paper companion.** This repository implements MACABEU, the adaptive defender
> introduced in *"Fight Fire with Fire: Adaptive Black-Box Defences for
> Misinformation Detection"* (M. Farré, P. Przybyła; under review at EMNLP 2026).
> The pre-trained policies used in every experiment of the paper are released
> under [`models/`](models/) (offline) and
> [`results/online/`](results/online/) (online).

MACABEU is a reinforcement learning agent that dynamically selects which preprocessing defense to apply to each input text at inference time, protecting NLP classifiers against adversarial attacks. It uses a **contextual bandit** formulation: a lightweight Q-network maps 10 statistical text features to one of 8 defense actions, choosing the best defense per input.

MACABEU is the defense counterpart to [XARELLO](https://aclanthology.org/2024.wassa-1.11/) (RL for adaptive attack). While XARELLO learns *which parts of the sentence to perturb*, MACABEU learns *which defense to apply* per example.

The full evaluation covers all **12 task–victim combinations** (4 BODEGA
tasks × 3 victim classifiers: BiLSTM, BERT, Gemma-2B) under **5 attackers**
(DeepWordBug, BERT-Attack, PWWS, Genetic, and XARELLO). The headline
XARELLO-vs-online-MACABEU sweep is complete on all 12 combinations; its
per-combo outputs (including HN-GEMMA) live in the
[`marti-farre/xarello`](https://github.com/marti-farre/xarello) repo under
`results/xarello_vs_macabeu_online/`. The 4 standard-attacker online policy
snapshots released in this repo cover 11 of the 12 combinations — see
"Released policies" below.

Built on top of the [BODEGA](https://doi.org/10.1017/nlp.2024.54) benchmark
(this paper uses the [`marti-farre/BODEGA`](https://github.com/marti-farre/BODEGA)
fork, which adds the static defence library MACABEU selects over) and uses
[`marti-farre/xarello`](https://github.com/marti-farre/xarello) as the adaptive
attacker.

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

1. Clone all three repos as siblings (the paper's fork of BODEGA + XARELLO):
   ```bash
   git clone https://github.com/marti-farre/BODEGA.git
   git clone https://github.com/marti-farre/macabeu.git
   git clone https://github.com/marti-farre/xarello.git   # optional, only for XARELLO eval
   ```

2. Set up the conda environment (shared with BODEGA / XARELLO). The versions
   below are the exact ones used to produce every number in the paper:
   ```bash
   conda create -n bodega python=3.10
   conda activate bodega
   conda install pytorch=2.7.1 pytorch-cuda=11.8 -c pytorch -c nvidia
   pip install "transformers==4.46.3" "tokenizers==0.20.3" "datasets==4.7.0"
   pip install "OpenAttack==2.1.1"
   pip install "bert-score==0.3.13" "editdistance==0.8.1"
   pip install "symspellpy==6.9.0" "homoglyphs==2.0.4"
   pip install git+https://github.com/lucadiliello/bleurt-pytorch.git   # 0.0.1
   pip install "peft==0.18.1" "bitsandbytes==0.49.2" "accelerate==1.13.0"  # Gemma victim only
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

## Released policies

The policies used in the paper are tracked in this repo so reviewers can
reproduce the headline numbers without re-running training:

| Path | Contents |
|------|----------|
| `models/{TASK}_{VICTIM}.pth` | Offline policy per (task, victim). 12 files. |
| `results/online/online_model_{TASK}_{VICTIM}_{ATTACKER}.pth` | Online policy snapshots from the 4 standard attackers (DeepWordBug, BERT-Attack, PWWS, Genetic). 44 files = 4 attackers × 11 task–victim combos; the HN-GEMMA online run for these 4 standard attackers was not executed (the headline XARELLO-vs-online-MACABEU sweep, which *was* run end-to-end for HN-GEMMA, is released in the `xarello` repo under `results/xarello_vs_macabeu_online/`). |

Each `.pth` file is ~16 KB; the full release is ~1.6 MB.
The matching `agent_data/*.npz` (raw per-example features + rewards used to
fit the offline Q-network) is **not** released — it can be regenerated from
the BODEGA attack splits via `runs/generate_defense_data.py`.

## Reproducing the paper

To reproduce the central RQ3 result (online MACABEU vs adaptive attacker
across all 12 task–victim combinations):

```bash
# 0. Sibling repos cloned + bodega conda env active + PYTHONPATH=../BODEGA:.
# 1. Train all 12 BODEGA victims (see ../BODEGA/runs/train_victims.py).
# 2. Evaluate offline MACABEU against XARELLO on a (task, victim) combo:
python runs/eval_defense_agent.py PR2 BiLSTM \
    ../BODEGA/data/PR2 ../BODEGA/data/PR2/BiLSTM-512.pth \
    models/PR2_BiLSTM.pth results/offline/PR2_BiLSTM/
# 3. Evaluate online MACABEU on the same combo (no pretrained policy needed):
python runs/eval_online.py PR2 BiLSTM \
    ../BODEGA/data/PR2 ../BODEGA/data/PR2/BiLSTM-512.pth \
    results/online/PR2/
```

The full sweep over (12 victims × 5 attackers × {offline, online}) is wired
into `scripts/run_all_tasks.sh`. Aggregated tables and figures are produced
by [`paper_assets/`](https://github.com/marti-farre/BODEGA/tree/main/paper_assets)
in the BODEGA fork.

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
├── models/                       #   Offline policy checkpoints (released, .pth tracked)
├── results/online/               #   Online policy checkpoints (released, .pth tracked)
└── agent_data/                   #   NPZ training data (gitignored, regenerable)
```

## Citation

```bibtex
@misc{farre2026fightfire,
  title  = {Fight Fire with Fire: Adaptive Black-Box Defences for Misinformation Detection},
  author = {Farr{\'e} Farr{\'u}s, Mart{\'\i} and Przyby{\l}a, Piotr},
  year   = {2026},
  note   = {Under review.}
}
```

## References

- **BODEGA**: [Verifying the Robustness of Automatic Credibility Assessment](https://doi.org/10.1017/nlp.2024.54) (NLP Journal). This repo uses the [`marti-farre/BODEGA`](https://github.com/marti-farre/BODEGA) fork that adds the static defence library.
- **XARELLO**: [Know Thine Enemy: Adaptive Attacks on Misinformation Detection Using Reinforcement Learning](https://aclanthology.org/2024.wassa-1.11/) (WASSA @ ACL 2024). Implementation at [`marti-farre/xarello`](https://github.com/marti-farre/xarello).
- Developed within the [ERINIA](https://www.upf.edu/web/erinia) project at the [TALN lab](https://www.upf.edu/web/taln/), Universitat Pompeu Fabra.
