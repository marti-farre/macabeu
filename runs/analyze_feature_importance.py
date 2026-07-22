"""Feature-importance analysis for MACABEU policies via mean-imputation.

Idea (per Piotr's suggestion, and matching reviewer #3's ask):
  1. Extract the 10-d state feature vector for every dev-split example.
  2. Compute the per-task feature means.
  3. For each trained policy, per example, per feature:
       a_orig     = policy.argmax_Q(features)
       a_ablated  = policy.argmax_Q(features with feature d replaced by mean)
       flip if a_orig != a_ablated
  4. Per-feature flip rate = fraction of examples where the decision flipped.
     High flip rate → policy relies on that feature.

Runs for all three MACABEU variants:
  offline  : models/{TASK}_{VICTIM}.pth (one policy per task-victim)
  oracle   : results/online/online_model_{TASK}_{VICTIM}_{ATK}.pth (per attacker)
  est_hard : results/online_true_hard/online_model_{TASK}_{VICTIM}_{ATK}.pth

Emits a long-form CSV so we can slice however we want later:
  policy_type,task,victim,attacker,feature,flip_rate,n_examples,n_flips

Requires BODEGA in PYTHONPATH.

Usage:
    PYTHONPATH=../BODEGA:. python runs/analyze_feature_importance.py \
        --data_root ../BODEGA/data \
        --out results/feature_importance.csv
"""

import argparse
import csv
import glob
import pathlib
import re
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from agent.features import TextFeatureExtractor
from agent.q_network import DefensePolicy
from agent.defense_env import DEFAULT_ACTION_SPACE, get_action_names
from utils.data_mappings import dataset_mapping, dataset_mapping_pairs


TASKS = ["PR2", "FC", "HN", "RD"]
VICTIMS = ["BiLSTM", "BERT", "GEMMA"]
FEATURE_NAMES = TextFeatureExtractor.FEATURE_NAMES

# Filename patterns for the three variants.
POLICY_PATTERNS = {
    "offline":  "models/{task}_{victim}.pth",
    "oracle":   "results/online/online_model_{task}_{victim}_*.pth",
    "est_hard": "results/online_true_hard/online_model_{task}_{victim}_*.pth",
}

# Extract attacker name from a filename like online_model_HN_BiLSTM_DeepWordBug.pth
ATK_RE = re.compile(r"online_model_[A-Z0-9]+_[A-Za-z]+_(?P<atk>[A-Za-z]+)\.pth$")


def read_texts_for_task(data_root: pathlib.Path, task: str):
    """Read the attack-split texts for a task without going through
    HuggingFace datasets (avoids needing a tokenizer/pretrained_model).
    Returns a list of raw text strings."""
    tsv = data_root / task / "attack.tsv"
    texts = []
    with tsv.open() as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            # BODEGA TSV: label \t source \t text (or with pairs joined by ~).
            texts.append(parts[-1])
    return texts


def load_policy(path: str, n_features: int, n_actions: int) -> DefensePolicy:
    policy = DefensePolicy(
        n_features=n_features, n_actions=n_actions,
        action_names=get_action_names(DEFAULT_ACTION_SPACE),
        warmup_steps=1,   # irrelevant for inference
    )
    policy.load(path)
    return policy


def actions_for_features(policy: DefensePolicy, features: np.ndarray) -> np.ndarray:
    """Batch-argmax the Q-net. features: (N, n_features). Returns (N,) int actions."""
    with torch.no_grad():
        feat_t = torch.tensor(features, dtype=torch.float32).to(policy.device)
        q = policy.q_net(feat_t)
        return q.argmax(dim=1).cpu().numpy()


def analyze_policy(policy: DefensePolicy, features_matrix: np.ndarray,
                   feature_means: np.ndarray) -> np.ndarray:
    """Per-feature flip rate for one policy. Returns array of shape (n_features,)."""
    orig = actions_for_features(policy, features_matrix)
    n_features = features_matrix.shape[1]
    flip_rates = np.zeros(n_features, dtype=np.float32)
    for d in range(n_features):
        ablated = features_matrix.copy()
        ablated[:, d] = feature_means[d]
        ablated_actions = actions_for_features(policy, ablated)
        flip_rates[d] = float(np.mean(orig != ablated_actions))
    return flip_rates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="../BODEGA/data",
                        help="BODEGA data directory (contains PR2/, FC/, HN/, RD/).")
    parser.add_argument("--macabeu_root", default=".",
                        help="Root of the macabeu repo (holds models/ and results/).")
    parser.add_argument("--out", default="results/feature_importance.csv")
    args = parser.parse_args()

    data_root = pathlib.Path(args.data_root)
    mac_root = pathlib.Path(args.macabeu_root)
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    extractor = TextFeatureExtractor()
    n_features = TextFeatureExtractor.NUM_FEATURES
    n_actions = len(DEFAULT_ACTION_SPACE)

    # 1) Extract feature matrices + per-task means once (cached).
    print("Extracting features per task...")
    per_task_features = {}
    per_task_means = {}
    for task in TASKS:
        texts = read_texts_for_task(data_root, task)
        if not texts:
            print(f"  [{task}] no texts found, skipping.")
            continue
        feats = np.stack([extractor.extract(t) for t in texts])
        per_task_features[task] = feats
        per_task_means[task] = feats.mean(axis=0)
        print(f"  [{task}] {len(texts)} texts, "
              f"feature-mean sample = {per_task_means[task][:3].round(3)}")

    # 2) Walk every policy file matching the three patterns.
    rows = []
    for variant, pattern in POLICY_PATTERNS.items():
        for task in TASKS:
            if task not in per_task_features:
                continue
            for victim in VICTIMS:
                glob_pattern = str(mac_root / pattern.format(task=task, victim=victim))
                paths = sorted(glob.glob(glob_pattern))
                if not paths:
                    continue
                for p in paths:
                    if variant == "offline":
                        attacker = "N/A"
                    else:
                        m = ATK_RE.search(p)
                        attacker = m.group("atk") if m else "unknown"
                    try:
                        policy = load_policy(p, n_features, n_actions)
                    except Exception as e:
                        print(f"  [SKIP] {p}: {e}")
                        continue
                    flip_rates = analyze_policy(policy, per_task_features[task],
                                                per_task_means[task])
                    n_ex = len(per_task_features[task])
                    for d, name in enumerate(FEATURE_NAMES):
                        rows.append({
                            "policy_type": variant,
                            "task":        task,
                            "victim":      victim,
                            "attacker":    attacker,
                            "feature":     name,
                            "flip_rate":   round(float(flip_rates[d]), 5),
                            "n_examples":  n_ex,
                            "n_flips":     int(round(flip_rates[d] * n_ex)),
                        })
                    print(f"  [{variant}] {task}/{victim}/{attacker}: "
                          f"top feature = {FEATURE_NAMES[int(np.argmax(flip_rates))]} "
                          f"({flip_rates.max():.3f})")

    # 3) Emit long-form CSV.
    fieldnames = ["policy_type", "task", "victim", "attacker",
                  "feature", "flip_rate", "n_examples", "n_flips"]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {out_path}")

    # 4) Console summary: aggregate flip rate per (variant, feature) over
    # everything, so we can eyeball the picture immediately.
    print("\nAverage flip rate per feature, aggregated across policies:")
    print(f"{'feature':<22} " + " ".join(f"{v:>10}" for v in POLICY_PATTERNS))
    per_variant = {v: {name: [] for name in FEATURE_NAMES} for v in POLICY_PATTERNS}
    for r in rows:
        per_variant[r["policy_type"]][r["feature"]].append(r["flip_rate"])
    for name in FEATURE_NAMES:
        cells = []
        for v in POLICY_PATTERNS:
            vals = per_variant[v][name]
            cells.append(f"{np.mean(vals):10.3f}" if vals else f"{'--':>10}")
        print(f"{name:<22} " + " ".join(cells))


if __name__ == "__main__":
    main()
