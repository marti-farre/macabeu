"""Evaluate clean accuracy of the RL defense selector on unattacked data.

Compares baseline (no defense) vs RL-selected defense accuracy to measure
the utility cost of the adaptive defense.

Requires BODEGA in PYTHONPATH.

Usage:
    PYTHONPATH=../BODEGA:. python runs/eval_clean_accuracy.py \
        PR2 BiLSTM ../BODEGA/data/PR2 ../BODEGA/data/PR2/BiLSTM-512.pth \
        models/PR2_BiLSTM.pth results/
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

from agent.defense_selector import RLDefenseSelector


def evaluate_accuracy(victim, texts, labels, batch_size=32):
    all_preds = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Evaluating", leave=False):
        batch_texts = texts[i:i + batch_size]
        batch_preds = victim.get_pred(batch_texts)
        all_preds.extend(batch_preds.tolist())
    predictions = np.array(all_preds)
    accuracy = np.mean(predictions == labels)
    return accuracy, predictions


def compute_f1(predictions, labels):
    TPs = np.sum((labels == 1) & (predictions == 1))
    FPs = np.sum((labels == 0) & (predictions == 1))
    FNs = np.sum((labels == 1) & (predictions == 0))
    if 2 * TPs + FPs + FNs == 0:
        return 0.0
    return 2 * TPs / (2 * TPs + FPs + FNs)


def main():
    parser = argparse.ArgumentParser(description='Evaluate RL defense clean accuracy')
    parser.add_argument('task', type=str, help='Task: PR2, FC, HN, RD')
    parser.add_argument('victim_model', type=str, help='Victim: BiLSTM, BERT, GEMMA')
    parser.add_argument('data_path', type=str)
    parser.add_argument('model_path', type=str)
    parser.add_argument('policy_path', type=str)
    parser.add_argument('output_dir', type=str, nargs='?', default=None)
    parser.add_argument('--subset', type=str, default='attack')

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

    # Load victim
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

    # Load data
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

    # Evaluate baseline (no defense)
    print("\nEvaluating baseline (no defense)...")
    base_acc, base_preds = evaluate_accuracy(base_victim, texts, labels)
    base_f1 = compute_f1(base_preds, labels)
    print(f"  Accuracy: {base_acc:.4f}, F1: {base_f1:.4f}")

    # Evaluate RL defense selector
    print(f"\nLoading RL policy from {policy_path}...")
    rl_victim = RLDefenseSelector(base_victim, str(policy_path))

    print("Evaluating RL defense selector...")
    rl_acc, rl_preds = evaluate_accuracy(rl_victim, texts, labels, batch_size=1)
    rl_f1 = compute_f1(rl_preds, labels)
    delta_acc = rl_acc - base_acc
    delta_f1 = rl_f1 - base_f1
    print(f"  Accuracy: {rl_acc:.4f}, F1: {rl_f1:.4f}")
    print(f"  Delta acc: {delta_acc:+.4f} ({delta_acc/base_acc*100:+.2f}%)")
    print(f"  Delta F1:  {delta_f1:+.4f} ({delta_f1/base_f1*100:+.2f}%)")

    # Action distribution
    stats = rl_victim.get_action_statistics()
    print("\nDefense action distribution on clean data:")
    for name, info in stats.items():
        print(f"  {name:<20s}: {info['count']:>5d} ({info['pct']:.1f}%)")

    # Save results
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_file = output_dir / f"clean_accuracy_{task}_{victim_model}_rl.txt"
        with open(out_file, 'w') as f:
            f.write(f"# RL Defense Clean Accuracy\n")
            f.write(f"Task: {task}\n")
            f.write(f"Victim: {victim_model}\n")
            f.write(f"Policy: {policy_path}\n")
            f.write(f"Subset: {args.subset}\n")
            f.write(f"Examples: {len(texts)}\n\n")
            f.write(f"# Baseline (no defense)\n")
            f.write(f"Accuracy: {base_acc:.4f}\n")
            f.write(f"F1: {base_f1:.4f}\n\n")
            f.write(f"# RL Defense Selector\n")
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
