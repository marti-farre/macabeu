"""Generate training data for the RL defense selector.

For each text in the attack set:
  1. Run each of 4 attackers -> get adversarial text
  2. Also include the clean (unattacked) version
  3. For each version, extract features and evaluate all 8 defenses
  4. Save full dataset as NPZ

Requires BODEGA in PYTHONPATH.

Usage:
    PYTHONPATH=../BODEGA:. python runs/generate_defense_data.py \
        PR2 BiLSTM ../BODEGA/data/PR2 ../BODEGA/data/PR2/BiLSTM-512.pth agent_data/
"""

import argparse
import os
import pathlib
import sys
import time
import random

import numpy as np
import torch
from datasets import Dataset

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

# BODEGA imports
import OpenAttack
from utils.data_mappings import dataset_mapping, dataset_mapping_pairs, SEPARATOR_CHAR
from utils.no_ssl_verify import no_ssl_verify
from victims.transformer import VictimTransformer, readfromfile_generator, PRETRAINED_BERT, PRETRAINED_GEMMA_2B
from victims.bilstm import VictimBiLSTM
from victims.unk_fix_wrapper import UNK_TEXT

# Macabeu imports
from agent.features import TextFeatureExtractor
from agent.defense_env import DefenseEnvironment, DEFAULT_ACTION_SPACE, get_action_names


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
    parser = argparse.ArgumentParser(description='Generate training data for RL defense selector')
    parser.add_argument('task', type=str, help='Task: PR2, FC, HN, RD')
    parser.add_argument('victim_model', type=str, help='Victim: BiLSTM, BERT, GEMMA')
    parser.add_argument('data_path', type=str, help='Path to task data')
    parser.add_argument('model_path', type=str, help='Path to trained victim model')
    parser.add_argument('output_dir', type=str, help='Output directory for NPZ files')
    parser.add_argument('--attackers', nargs='+', default=ATTACKERS,
                        help='Which attackers to use')
    parser.add_argument('--max_examples', type=int, default=0,
                        help='Max examples to process (0 = all)')
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
    victim, pretrained_model = load_victim(args.victim_model, model_path, task, device)

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

    # Filter to correctly classified examples (untargeted attack setting)
    dataset_list = [inst for inst in dataset if victim.get_pred([inst["x"]])[0] == inst["y"]]
    if args.max_examples > 0:
        dataset_list = dataset_list[:args.max_examples]
    print(f"Using {len(dataset_list)} correctly classified examples")

    # Setup defense environment
    print("Setting up defense environment...")
    defense_env = DefenseEnvironment(victim, seed=42)
    feature_extractor = TextFeatureExtractor()
    n_actions = len(DEFAULT_ACTION_SPACE)
    action_names = get_action_names()

    # Storage
    all_features = []
    all_rewards = []
    all_attack_types = []
    all_labels = []
    all_texts = []

    # Phase 1: Clean examples
    print("\n=== Evaluating clean examples ===")
    for i, inst in enumerate(dataset_list):
        text = inst["x"]
        label = inst["y"]
        features = feature_extractor.extract(text)
        rewards = defense_env.evaluate_all_defenses(text, label)

        all_features.append(features)
        all_rewards.append(rewards)
        all_attack_types.append('clean')
        all_labels.append(label)
        all_texts.append(text)

        if (i + 1) % 50 == 0:
            print(f"  Clean: {i+1}/{len(dataset_list)}")

    # Phase 2: Adversarial examples from each attacker
    for attack_name in args.attackers:
        print(f"\n=== Generating adversarial examples with {attack_name} ===")
        start_time = time.time()

        with no_ssl_verify():
            attacker = create_attacker(attack_name, device)

        success_count = 0
        for i, inst in enumerate(dataset_list):
            text = inst["x"]
            label = inst["y"]

            # Generate adversarial example
            try:
                goal = OpenAttack.attack_assist.goal.ClassifierGoal(label, False)
                victim.set_context(inst, None)
                try:
                    adv_text = attacker(victim, inst)
                finally:
                    victim.clear_context()
            except Exception as e:
                print(f"  Attack failed on example {i}: {e}")
                adv_text = None

            if adv_text is not None:
                success_count += 1
                features = feature_extractor.extract(adv_text)
                rewards = defense_env.evaluate_all_defenses(adv_text, label)

                all_features.append(features)
                all_rewards.append(rewards)
                all_attack_types.append(attack_name)
                all_labels.append(label)
                all_texts.append(adv_text)

            if (i + 1) % 50 == 0:
                elapsed = time.time() - start_time
                print(f"  {attack_name}: {i+1}/{len(dataset_list)} "
                      f"({success_count} successful, {elapsed:.0f}s)")

        elapsed = time.time() - start_time
        print(f"  {attack_name} done: {success_count}/{len(dataset_list)} successful "
              f"in {elapsed:.0f}s")

        # Free attacker memory
        del attacker

    # Save dataset
    features_arr = np.array(all_features, dtype=np.float32)
    rewards_arr = np.array(all_rewards, dtype=np.float32)
    labels_arr = np.array(all_labels, dtype=np.int32)

    out_file = output_dir / f"{task}_{args.victim_model}.npz"
    np.savez(
        out_file,
        features=features_arr,
        rewards=rewards_arr,
        labels=labels_arr,
        attack_types=np.array(all_attack_types),
        action_names=np.array(action_names),
    )

    print(f"\n=== Dataset saved to {out_file} ===")
    print(f"Total examples: {len(all_features)}")
    print(f"  Clean: {sum(1 for a in all_attack_types if a == 'clean')}")
    for atk in args.attackers:
        count = sum(1 for a in all_attack_types if a == atk)
        print(f"  {atk}: {count}")
    print(f"Features shape: {features_arr.shape}")
    print(f"Rewards shape: {rewards_arr.shape}")

    # Print oracle analysis: for each attack type, which defense is best on average?
    print("\n=== Oracle analysis (best defense per attack type) ===")
    for atk_type in ['clean'] + list(args.attackers):
        mask = np.array([a == atk_type for a in all_attack_types])
        if mask.sum() == 0:
            continue
        avg_rewards = rewards_arr[mask].mean(axis=0)
        best_action = avg_rewards.argmax()
        print(f"  {atk_type:15s}: best={action_names[best_action]:20s} "
              f"(avg reward={avg_rewards[best_action]:.3f})")
        for j, name in enumerate(action_names):
            print(f"    {name:20s}: {avg_rewards[j]:.3f}")


if __name__ == '__main__':
    main()
