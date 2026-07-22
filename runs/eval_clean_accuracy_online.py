"""Evaluate clean accuracy of the ONLINE RL defense selector on unattacked data.

Same online selector as eval_online.py, but there is no attacker. We walk the
clean dev split, let the policy update after each example, and measure the
final accuracy. Simulates "deployment on benign traffic": how much does an
adaptive online defender cost (or save) when nothing is under attack?

Supports both label sources:
  --label_source oracle   → reward uses the gold label (MACABEU-Oracle)
  --label_source mv7      → reward uses a MajorityVote-7 pseudo-label (MACABEU-est)

Requires BODEGA in PYTHONPATH.

Usage:
    PYTHONPATH=../BODEGA:. python runs/eval_clean_accuracy_online.py \
        PR2 BiLSTM ../BODEGA/data/PR2 ../BODEGA/data/PR2/BiLSTM-512.pth \
        models/PR2_BiLSTM.pth results/clean_accuracy_online/ \
        --label_source mv7 --reward_mode hard
"""

import argparse
import pathlib
import sys

import numpy as np
import torch
from datasets import Dataset
from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from victims.transformer import VictimTransformer, readfromfile_generator, PRETRAINED_BERT, PRETRAINED_GEMMA_2B
from victims.bilstm import VictimBiLSTM
from utils.data_mappings import dataset_mapping, dataset_mapping_pairs

from agent.online_selector import OnlineRLDefenseSelector


def compute_f1(predictions, labels):
    TPs = np.sum((labels == 1) & (predictions == 1))
    FPs = np.sum((labels == 0) & (predictions == 1))
    FNs = np.sum((labels == 1) & (predictions == 0))
    if 2 * TPs + FPs + FNs == 0:
        return 0.0
    return 2 * TPs / (2 * TPs + FPs + FNs)


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate ONLINE RL defense clean accuracy (adaptive on clean traffic)')
    parser.add_argument('task', type=str, help='Task: PR2, FC, HN, RD')
    parser.add_argument('victim_model', type=str, help='Victim: BiLSTM, BERT, GEMMA')
    parser.add_argument('data_path', type=str)
    parser.add_argument('model_path', type=str)
    parser.add_argument('policy_path', type=str,
                        help='Offline-trained MACABEU policy used as warm start')
    parser.add_argument('output_dir', type=str, nargs='?', default=None)
    parser.add_argument('--subset', type=str, default='attack',
                        help='Which split to evaluate on (default: attack, matches BODEGA convention)')
    parser.add_argument('--label_source', type=str, default='oracle',
                        choices=['oracle', 'mv7'],
                        help="Reward label source. 'oracle' = gold label; "
                             "'mv7' = MajorityVote-7 pseudo-label.")
    parser.add_argument('--reward_mode', type=str, default='hard',
                        choices=['hard', 'soft'])
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max_eps', type=float, default=1.0)
    parser.add_argument('--min_eps', type=float, default=0.05)
    parser.add_argument('--warmup', type=int, default=50)

    args = parser.parse_args()
    task = args.task
    victim_model = args.victim_model
    data_path = pathlib.Path(args.data_path)
    model_path = pathlib.Path(args.model_path)
    policy_path = pathlib.Path(args.policy_path)
    output_dir = pathlib.Path(args.output_dir) if args.output_dir else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with_pairs = (task == 'FC' or task == 'C19')

    print(f"Loading {victim_model} model...")
    if victim_model == 'BiLSTM':
        pretrained_model = PRETRAINED_BERT
        base_victim = VictimBiLSTM(model_path, task, device)
    elif victim_model == 'BERT':
        pretrained_model = PRETRAINED_BERT
        base_victim = VictimTransformer(model_path, task, pretrained_model, False, device)
    elif victim_model == 'GEMMA':
        pretrained_model = PRETRAINED_GEMMA_2B
        base_victim = VictimTransformer(model_path, task, pretrained_model, True, device)
    else:
        raise ValueError(f"Unknown victim: {victim_model}")

    print(f"Loading {args.subset} data...")
    test_dataset = Dataset.from_generator(
        readfromfile_generator,
        gen_kwargs={'subset': args.subset, 'dir': data_path,
                    'pretrained_model': pretrained_model, 'trim_text': True,
                    'with_pairs': with_pairs}
    )
    if not with_pairs:
        dataset = test_dataset.map(function=dataset_mapping)
    else:
        dataset = test_dataset.map(function=dataset_mapping_pairs)

    texts = [item['x'] for item in dataset]
    labels = np.array([item['y'] for item in dataset])
    print(f"Loaded {len(texts)} examples")

    # Baseline (no defence): batch-eval the base victim.
    print("\nEvaluating baseline (no defense)...")
    all_preds = []
    for i in tqdm(range(0, len(texts), 32), desc="Baseline", leave=False):
        all_preds.extend(base_victim.get_pred(texts[i:i + 32]).tolist())
    base_preds = np.array(all_preds)
    base_acc = float(np.mean(base_preds == labels))
    base_f1 = compute_f1(base_preds, labels)
    print(f"  Accuracy: {base_acc:.4f}, F1: {base_f1:.4f}")

    print(f"\nLoading ONLINE policy from {policy_path}...")
    print(f"  label_source={args.label_source}, reward_mode={args.reward_mode}")
    online_victim = OnlineRLDefenseSelector(
        base_victim, seed=42,
        lr=args.lr,
        max_eps=args.max_eps, min_eps=args.min_eps,
        warmup_examples=args.warmup,
        pretrained_path=str(policy_path),
        label_source=args.label_source,
        reward_mode=args.reward_mode,
        verbose=False,
    )

    # Online eval: per-example, so the policy sees each input and updates after.
    print("Evaluating ONLINE RL defense on clean traffic (with adaptation)...")
    rl_preds = []
    for i in tqdm(range(len(texts)), desc="Online", leave=False):
        pred = int(online_victim.get_pred([texts[i]])[0])
        rl_preds.append(pred)
        # Feed the reward signal. For label_source='mv7' the true_label is ignored
        # internally (the selector computes its own pseudo-label from the last-seen text).
        online_victim.observe_result(int(labels[i]), pred)
    rl_preds = np.array(rl_preds)
    rl_acc = float(np.mean(rl_preds == labels))
    rl_f1 = compute_f1(rl_preds, labels)
    delta_acc = rl_acc - base_acc
    delta_f1 = rl_f1 - base_f1
    print(f"  Accuracy: {rl_acc:.4f}, F1: {rl_f1:.4f}")
    print(f"  Delta acc: {delta_acc:+.4f} ({delta_acc/base_acc*100:+.2f}%)")
    print(f"  Delta F1:  {delta_f1:+.4f} ({delta_f1/base_f1*100:+.2f}%)")

    stats = online_victim.get_action_statistics()
    print("\nDefense action distribution on clean data:")
    for name, info in stats.items():
        print(f"  {name:<20s}: {info['count']:>5d} ({info['pct']:.1f}%)")

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"{args.label_source}_{args.reward_mode}"
        out_file = output_dir / f"clean_accuracy_{task}_{victim_model}_online_{suffix}.txt"
        with open(out_file, 'w') as f:
            f.write(f"# Online RL Defense Clean Accuracy\n")
            f.write(f"Task: {task}\n")
            f.write(f"Victim: {victim_model}\n")
            f.write(f"Policy (warm start): {policy_path}\n")
            f.write(f"Subset: {args.subset}\n")
            f.write(f"Label source: {args.label_source}\n")
            f.write(f"Reward mode: {args.reward_mode}\n")
            f.write(f"Examples: {len(texts)}\n\n")
            f.write(f"# Baseline (no defense)\n")
            f.write(f"Accuracy: {base_acc:.4f}\n")
            f.write(f"F1: {base_f1:.4f}\n\n")
            f.write(f"# Online RL Defense Selector (adaptive on clean traffic)\n")
            f.write(f"Accuracy: {rl_acc:.4f}\n")
            f.write(f"F1: {rl_f1:.4f}\n")
            f.write(f"Delta accuracy: {delta_acc:+.4f} ({delta_acc/base_acc*100:+.2f}%)\n")
            f.write(f"Delta F1: {delta_f1:+.4f} ({delta_f1/base_f1*100:+.2f}%)\n\n")
            f.write(f"# Action Distribution\n")
            for name, info in stats.items():
                f.write(f"{name:<20s}: {info['count']:>5d} ({info['pct']:.1f}%)\n")
        print(f"\nSaved to {out_file}")


if __name__ == '__main__':
    main()
