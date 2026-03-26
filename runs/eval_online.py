"""Online RL defense evaluation: the agent learns DURING the attack.

Unlike eval_defense_agent.py (frozen policy), this script lets the
agent update its Q-network after each attacked example. The agent
starts uncertain and progressively adapts to the attacker's strategy.

Requires BODEGA in PYTHONPATH.

Usage:
    PYTHONPATH=../BODEGA:. python runs/eval_online.py \
        PR2 BiLSTM ../BODEGA/data/PR2 ../BODEGA/data/PR2/BiLSTM-512.pth results/online/ \
        --attackers DeepWordBug BERTattack PWWS Genetic
"""

import argparse
import gc
import os
import pathlib
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
from agent.online_selector import OnlineRLDefenseSelector


ATTACKERS = ['DeepWordBug', 'BERTattack', 'PWWS', 'Genetic']


def create_attacker(attack_name, device):
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
    if victim_model == 'BiLSTM':
        return VictimBiLSTM(model_path, task, device), PRETRAINED_BERT
    elif victim_model == 'BERT':
        return VictimTransformer(model_path, task, PRETRAINED_BERT, False, device), PRETRAINED_BERT
    elif victim_model == 'GEMMA':
        return VictimTransformer(model_path, task, PRETRAINED_GEMMA_2B, True, device), PRETRAINED_GEMMA_2B
    else:
        raise ValueError(f"Unknown victim: {victim_model}")


def run_online_attack(attacker, online_victim, dataset_list, task, device,
                      semantic_scorer='BERTscore'):
    """Run attack with online learning: update agent after each example."""

    scorer = BODEGAScore(device, task, align_sentences=True,
                         semantic_scorer=semantic_scorer)

    n_success = 0
    n_total = 0
    total_queries = 0
    start_time = time.time()

    for i, inst in enumerate(dataset_list):
        x_orig = inst["x"]
        true_label = inst["y"]
        n_total += 1

        # Check if victim classifies correctly before attack
        orig_pred = online_victim.get_pred([x_orig])[0]
        if orig_pred != true_label:
            # Skip misclassified (consistent with BODEGA eval)
            continue

        # Run the attack
        try:
            goal = OpenAttack.attack_assist.goal.ClassifierGoal(true_label, False)
            online_victim.set_context(inst, None)
            try:
                adv_text = attacker(online_victim, inst)
                queries = online_victim.context.invoke
            finally:
                online_victim.clear_context()
        except Exception as e:
            adv_text = None
            queries = 0

        total_queries += queries

        if adv_text is not None:
            # Attack produced an adversarial example
            final_pred = online_victim.get_pred([adv_text])[0]
            attack_succeeded = (final_pred != true_label)

            if attack_succeeded:
                n_success += 1

            # Feed to scorer for BODEGA computation
            # after_attack expects (input_dict, adversarial_string_or_None)
            if attack_succeeded:
                scorer.after_attack({"x": x_orig, "y": true_label}, adv_text)
            else:
                scorer.after_attack({"x": x_orig, "y": true_label}, None)

            # ONLINE LEARNING: observe result and update
            online_victim.observe_result(true_label, final_pred)
        else:
            # Attack failed to produce adversarial example
            scorer.after_attack({"x": x_orig, "y": true_label}, None)

            # Observe success (defense held)
            online_victim.observe_result(true_label, true_label)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            asr = n_success / max(n_total, 1)
            avg_reward = np.mean(online_victim.reward_history[-50:]) if online_victim.reward_history else 0
            print(f"  [{i+1}/{len(dataset_list)}] ASR={asr:.3f} | "
                  f"eps={online_victim.policy.eps:.3f} | "
                  f"avg_reward(50)={avg_reward:.3f} | "
                  f"{elapsed:.0f}s")

    attack_time = time.time() - start_time

    # Compute BODEGA scores
    score_success, score_semantic, score_character, score_BODEGA = scorer.compute()
    avg_queries = total_queries / max(n_total, 1)

    return {
        'success': score_success,
        'semantic': score_semantic,
        'character': score_character,
        'BODEGA': score_BODEGA,
        'queries': avg_queries,
        'time': attack_time,
        'n_total': n_total,
        'n_success': n_success,
    }


def main():
    parser = argparse.ArgumentParser(description='Online RL defense evaluation')
    parser.add_argument('task', type=str)
    parser.add_argument('victim_model', type=str)
    parser.add_argument('data_path', type=str)
    parser.add_argument('model_path', type=str, help='Path to victim model')
    parser.add_argument('output_dir', type=str)
    parser.add_argument('--attackers', nargs='+', default=ATTACKERS)
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained offline policy (warm start)')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max_eps', type=float, default=1.0,
                        help='Initial exploration rate')
    parser.add_argument('--min_eps', type=float, default=0.05)
    parser.add_argument('--warmup', type=int, default=50,
                        help='Examples before epsilon reaches min')
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

    # Load dataset
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
    dataset_list = list(dataset)

    print(f"Dataset size: {len(dataset_list)}")

    results = {}
    for attack_name in args.attackers:
        print(f"\n{'='*60}")
        print(f"ONLINE RL vs {attack_name}")
        print(f"{'='*60}")

        # Fresh online agent for each attacker
        online_victim = OnlineRLDefenseSelector(
            base_victim,
            seed=42,
            lr=args.lr,
            max_eps=args.max_eps,
            min_eps=args.min_eps,
            warmup_examples=args.warmup,
            pretrained_path=args.pretrained,
            verbose=False
        )

        with no_ssl_verify():
            attacker = create_attacker(attack_name, device)

        with no_ssl_verify():
            res = run_online_attack(
                attacker, online_victim, dataset_list, task, device,
                semantic_scorer=args.semantic_scorer
            )

        results[attack_name] = res
        action_stats = online_victim.get_action_statistics()

        print(f"\nResults for {attack_name}:")
        print(f"  Success:   {res['success']:.4f}")
        print(f"  Semantic:  {res['semantic']:.4f}")
        print(f"  Character: {res['character']:.4f}")
        print(f"  BODEGA:    {res['BODEGA']:.4f}")
        print(f"  Queries:   {res['queries']:.1f}")
        print(f"  Time:      {res['time']:.1f}s")
        print(f"  Defense selections:")
        for name, stats in action_stats.items():
            if stats['count'] > 0:
                print(f"    {name:20s}: {stats['count']:5d} ({stats['pct']:.1f}%)")

        # Learning curve
        curve = online_victim.get_learning_curve(window=20)
        if curve:
            print(f"  Learning curve (reward, window=20):")
            print(f"    Start: {curve[0]:.3f}")
            print(f"    End:   {curve[-1]:.3f}")
            print(f"    Peak:  {max(curve):.3f}")

        # Save results
        out_file = output_dir / f"results_{task}_False_{attack_name}_{args.victim_model}_online_rl.txt"
        with open(out_file, 'w') as f:
            f.write("# Experiment Configuration\n")
            f.write(f"Task: {task}\n")
            f.write(f"Attack: {attack_name}\n")
            f.write(f"Victim: {args.victim_model}\n")
            f.write(f"Defense: online_rl\n")
            f.write(f"LR: {args.lr}\n")
            f.write(f"Max eps: {args.max_eps}\n")
            f.write(f"Min eps: {args.min_eps}\n")
            f.write(f"Warmup: {args.warmup}\n")
            f.write(f"Pretrained: {args.pretrained or 'none'}\n")
            f.write(f"Semantic scorer: {args.semantic_scorer}\n")
            f.write(f"\n# Results\n")
            f.write(f"Subset size: {res['n_total']}\n")
            f.write(f"Success score: {res['success']}\n")
            f.write(f"Semantic score: {res['semantic']}\n")
            f.write(f"Character score: {res['character']}\n")
            f.write(f"BODEGA score: {res['BODEGA']}\n")
            f.write(f"Queries per example: {res['queries']}\n")
            f.write(f"Total attack time: {res['time']}\n")
            f.write(f"\n# Defense Action Distribution\n")
            for name, stats in action_stats.items():
                f.write(f"{name}: {stats['count']} ({stats['pct']:.1f}%)\n")
            f.write(f"\n# Learning Curve (window=20)\n")
            for j, val in enumerate(curve):
                f.write(f"{j},{val:.4f}\n")

        # Save trained online model
        model_out = output_dir / f"online_model_{task}_{args.victim_model}_{attack_name}.pth"
        online_victim.save(str(model_out))

        del attacker, online_victim
        gc.collect()

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY: Online RL Defense on {task}/{args.victim_model}")
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


if __name__ == '__main__':
    main()
