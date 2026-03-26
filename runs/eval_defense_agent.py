"""Evaluate the trained RL defense selector against static defenses.

Runs the full OpenAttack attack pipeline using the RL selector as the
defense wrapper, then compares BODEGA scores against fixed defenses.

Requires BODEGA in PYTHONPATH.

Usage:
    PYTHONPATH=../BODEGA:. python runs/eval_defense_agent.py \
        PR2 BiLSTM ../BODEGA/data/PR2 ../BODEGA/data/PR2/BiLSTM-512.pth \
        models/PR2_BiLSTM.pth results/
"""

import argparse
import gc
import os
import pathlib
import sys
import time
import random

import numpy as np
import torch
from datasets import Dataset

random.seed(10)
torch.manual_seed(10)
np.random.seed(0)

# BODEGA imports
import OpenAttack
from metrics.BODEGAScore import BODEGAScore
from utils.data_mappings import dataset_mapping, dataset_mapping_pairs, SEPARATOR_CHAR
from utils.no_ssl_verify import no_ssl_verify
from victims.transformer import VictimTransformer, readfromfile_generator, PRETRAINED_BERT, PRETRAINED_GEMMA_2B
from victims.bilstm import VictimBiLSTM
from victims.unk_fix_wrapper import UNK_TEXT

# Macabeu imports
from agent.defense_selector import RLDefenseSelector


ATTACKERS = ['DeepWordBug', 'BERTattack', 'PWWS', 'Genetic']


def create_attacker(attack_name, device):
    """Create an OpenAttack attacker by name."""
    filter_words = OpenAttack.attack_assist.filter_words.get_default_filter_words('english') + [SEPARATOR_CHAR]
    with no_ssl_verify():
        if attack_name == 'DeepWordBug':
            return OpenAttack.attackers.DeepWordBugAttacker(token_unk=UNK_TEXT)
        elif attack_name == 'BERTattack':
            return OpenAttack.attackers.BERTAttacker(filter_words=filter_words, use_bpe=False, device=device)
        elif attack_name == 'PWWS':
            return OpenAttack.attackers.PWWSAttacker(token_unk=UNK_TEXT, lang='english', filter_words=filter_words)
        elif attack_name == 'Genetic':
            return OpenAttack.attackers.GeneticAttacker(lang='english', filter_words=filter_words)
        else:
            raise ValueError(f"Unknown attacker: {attack_name}")


def load_victim(victim_model, model_path, task, device):
    """Load a BODEGA victim model."""
    if victim_model == 'BiLSTM':
        return VictimBiLSTM(model_path, task, device), PRETRAINED_BERT
    elif victim_model == 'BERT':
        return VictimTransformer(model_path, task, PRETRAINED_BERT, False, device), PRETRAINED_BERT
    elif victim_model == 'GEMMA':
        return VictimTransformer(model_path, task, PRETRAINED_GEMMA_2B, True, device), PRETRAINED_GEMMA_2B
    else:
        raise ValueError(f"Unknown victim: {victim_model}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate RL defense selector')
    parser.add_argument('task', type=str)
    parser.add_argument('victim_model', type=str)
    parser.add_argument('data_path', type=str)
    parser.add_argument('model_path', type=str, help='Path to victim model')
    parser.add_argument('policy_path', type=str, help='Path to trained RL policy')
    parser.add_argument('output_dir', type=str)
    parser.add_argument('--attackers', nargs='+', default=ATTACKERS)
    parser.add_argument('--semantic_scorer', type=str, default='BERTscore',
                        choices=['BERTscore', 'BLEURT'])
    args = parser.parse_args()

    task = args.task
    data_path = pathlib.Path(args.data_path)
    model_path = pathlib.Path(args.model_path)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with_pairs = (task in ['FC', 'C19'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load victim
    print(f"Loading {args.victim_model} victim for {task}...")
    base_victim, pretrained_model = load_victim(args.victim_model, model_path, task, device)

    # Wrap with RL defense selector
    print(f"Loading RL defense policy from {args.policy_path}...")
    victim = RLDefenseSelector(
        base_victim,
        policy_path=args.policy_path,
        seed=42,
        verbose=False
    )

    # Load attack dataset
    print("Loading attack dataset...")
    test_dataset = Dataset.from_generator(
        readfromfile_generator,
        gen_kwargs={'subset': 'attack', 'dir': data_path,
                    'pretrained_model': pretrained_model, 'trim_text': True,
                    'with_pairs': with_pairs}
    )
    if not with_pairs:
        dataset = test_dataset.map(function=dataset_mapping)
        dataset = dataset.remove_columns(["text"])
    else:
        dataset = test_dataset.map(function=dataset_mapping_pairs)
        dataset = dataset.remove_columns(["text1", "text2"])
    dataset = dataset.remove_columns(["fake"])

    print(f"Dataset size: {len(dataset)}")

    results = {}
    for attack_name in args.attackers:
        print(f"\n{'='*50}")
        print(f"Running {attack_name} against RL defense selector")
        print(f"{'='*50}")

        # Reset action counts
        victim.action_counts[:] = 0

        # Create attacker
        with no_ssl_verify():
            attacker = create_attacker(attack_name, device)

        # Run attack evaluation
        scorer = BODEGAScore(device, task, align_sentences=True,
                             semantic_scorer=args.semantic_scorer)
        attack_eval = OpenAttack.AttackEval(
            attacker, victim, language='english',
            metrics=[scorer]
        )

        start = time.time()
        with no_ssl_verify():
            summary = attack_eval.eval(dataset, visualize=True, progress_bar=False)
        attack_time = time.time() - start

        # Compute scores
        start = time.time()
        score_success, score_semantic, score_character, score_BODEGA = scorer.compute()
        eval_time = time.time() - start

        # Get action statistics
        action_stats = victim.get_action_statistics()

        results[attack_name] = {
            'success': score_success,
            'semantic': score_semantic,
            'character': score_character,
            'BODEGA': score_BODEGA,
            'queries': summary['Avg. Victim Model Queries'],
            'time': attack_time,
            'action_stats': action_stats,
        }

        # Print results
        print(f"\nResults for {attack_name}:")
        print(f"  Success:   {score_success:.4f}")
        print(f"  Semantic:  {score_semantic:.4f}")
        print(f"  Character: {score_character:.4f}")
        print(f"  BODEGA:    {score_BODEGA:.4f}")
        print(f"  Queries:   {summary['Avg. Victim Model Queries']:.1f}")
        print(f"  Time:      {attack_time:.1f}s")
        print(f"  Defense selections:")
        for name, stats in action_stats.items():
            if stats['count'] > 0:
                print(f"    {name:20s}: {stats['count']:5d} ({stats['pct']:.1f}%)")

        # Save per-attack results
        out_file = output_dir / f"results_{task}_False_{attack_name}_{args.victim_model}_rl_defense.txt"
        with open(out_file, 'w') as f:
            f.write("# Experiment Configuration\n")
            f.write(f"Task: {task}\n")
            f.write(f"Attack: {attack_name}\n")
            f.write(f"Victim: {args.victim_model}\n")
            f.write(f"Defense: rl_defense\n")
            f.write(f"Semantic scorer: {args.semantic_scorer}\n")
            f.write(f"\n# Results\n")
            f.write(f"Subset size: {len(dataset)}\n")
            f.write(f"Success score: {score_success}\n")
            f.write(f"Semantic score: {score_semantic}\n")
            f.write(f"Character score: {score_character}\n")
            f.write(f"BODEGA score: {score_BODEGA}\n")
            f.write(f"Queries per example: {summary['Avg. Victim Model Queries']}\n")
            f.write(f"Total attack time: {attack_time}\n")
            f.write(f"Total evaluation time: {eval_time}\n")
            f.write(f"\n# Defense Action Distribution\n")
            for name, stats in action_stats.items():
                f.write(f"{name}: {stats['count']} ({stats['pct']:.1f}%)\n")

        # Cleanup
        del attacker
        gc.collect()

    # Print summary table
    print(f"\n{'='*70}")
    print(f"SUMMARY: RL Defense Selector on {task}/{args.victim_model}")
    print(f"{'='*70}")
    print(f"{'Attack':15s} {'Success':>8s} {'Semantic':>9s} {'Char':>8s} {'BODEGA':>8s} {'Queries':>8s}")
    print("-" * 70)
    avg_bodega = 0.0
    for atk in args.attackers:
        if atk in results:
            r = results[atk]
            print(f"{atk:15s} {r['success']:8.4f} {r['semantic']:9.4f} "
                  f"{r['character']:8.4f} {r['BODEGA']:8.4f} {r['queries']:8.1f}")
            avg_bodega += r['BODEGA']
    avg_bodega /= max(len(results), 1)
    print("-" * 70)
    print(f"{'Average':15s} {'':8s} {'':9s} {'':8s} {avg_bodega:8.4f}")

    # Save summary
    summary_file = output_dir / f"summary_{task}_{args.victim_model}_rl_defense.txt"
    with open(summary_file, 'w') as f:
        f.write(f"RL Defense Selector Summary: {task}/{args.victim_model}\n\n")
        f.write(f"{'Attack':15s} {'Success':>8s} {'Semantic':>9s} {'Char':>8s} {'BODEGA':>8s}\n")
        for atk in args.attackers:
            if atk in results:
                r = results[atk]
                f.write(f"{atk:15s} {r['success']:8.4f} {r['semantic']:9.4f} "
                        f"{r['character']:8.4f} {r['BODEGA']:8.4f}\n")
        f.write(f"\nAvg BODEGA: {avg_bodega:.4f}\n")


if __name__ == '__main__':
    main()
