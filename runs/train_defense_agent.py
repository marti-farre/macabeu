"""Train the RL defense selector agent on pre-generated data.

Loads the NPZ file from generate_defense_data.py and trains a Q-network
using experience replay (contextual bandit formulation, gamma=0).

Usage:
    PYTHONPATH=../BODEGA:. python runs/train_defense_agent.py \
        agent_data/PR2_BiLSTM.npz models/PR2_BiLSTM.pth
"""

import argparse
import pathlib
import random

import numpy as np
import torch

from agent.q_network import DefensePolicy
from agent.replay_buffer import ReplayBuffer

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)


def main():
    parser = argparse.ArgumentParser(description='Train RL defense selector')
    parser.add_argument('data_path', type=str, help='Path to NPZ training data')
    parser.add_argument('model_path', type=str, help='Output path for trained model')
    parser.add_argument('--epochs', type=int, default=30, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Minibatch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--buffer_size', type=int, default=10000, help='Replay buffer size')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split')
    parser.add_argument('--warmup_steps', type=int, default=500, help='Epsilon decay steps')
    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.data_path}...")
    data = np.load(args.data_path, allow_pickle=True)
    features = data['features']  # (N, 10)
    rewards = data['rewards']    # (N, 8)
    attack_types = data['attack_types']  # (N,)
    action_names = list(data['action_names'])

    n_examples, n_features = features.shape
    n_actions = rewards.shape[1]
    print(f"Dataset: {n_examples} examples, {n_features} features, {n_actions} actions")
    print(f"Actions: {action_names}")

    # Train/val split
    indices = np.arange(n_examples)
    np.random.shuffle(indices)
    val_size = int(n_examples * args.val_split)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    train_features = features[train_idx]
    train_rewards = rewards[train_idx]
    val_features = features[val_idx]
    val_rewards = rewards[val_idx]

    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")

    # Initialize policy and replay buffer
    policy = DefensePolicy(
        n_features=n_features,
        n_actions=n_actions,
        action_names=action_names,
        lr=args.lr,
        warmup_steps=args.warmup_steps
    )
    replay_buffer = ReplayBuffer(capacity=args.buffer_size)

    # Output path
    model_path = pathlib.Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_reward = -float('inf')
    best_epoch = 0

    print(f"\nTraining for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        # Shuffle training data
        perm = np.random.permutation(len(train_features))
        epoch_loss = 0.0
        n_updates = 0

        for i in perm:
            feat = train_features[i]
            reward_vec = train_rewards[i]

            # Select action (epsilon-greedy)
            action = policy.select_action(feat, greedy=False)
            reward = reward_vec[action]

            # Store experience
            replay_buffer.push(feat, action, reward)

            # Learn from minibatch
            if len(replay_buffer) >= args.batch_size:
                f_batch, a_batch, r_batch = replay_buffer.sample(args.batch_size)
                loss = policy.update(f_batch, a_batch, r_batch)
                epoch_loss += loss
                n_updates += 1

        avg_loss = epoch_loss / max(n_updates, 1)

        # Validation: greedy policy, measure average reward
        val_total_reward = 0.0
        val_action_counts = np.zeros(n_actions, dtype=int)
        for i in range(len(val_features)):
            action = policy.select_action(val_features[i], greedy=True)
            val_action_counts[action] += 1
            val_total_reward += val_rewards[i, action]

        avg_val_reward = val_total_reward / max(len(val_features), 1)

        # Compare with oracle (always pick best defense per example)
        oracle_reward = val_rewards.max(axis=1).mean()

        # Compare with best fixed defense
        fixed_rewards = val_rewards.mean(axis=0)
        best_fixed_idx = fixed_rewards.argmax()
        best_fixed_reward = fixed_rewards[best_fixed_idx]

        if avg_val_reward > best_val_reward:
            best_val_reward = avg_val_reward
            best_epoch = epoch
            policy.save(str(model_path))

        print(f"  Epoch {epoch+1:3d} | loss={avg_loss:.4f} | eps={policy.eps:.3f} | "
              f"val_reward={avg_val_reward:.3f} | oracle={oracle_reward:.3f} | "
              f"best_fixed={best_fixed_reward:.3f} ({action_names[best_fixed_idx]})")

        # Print action distribution
        if (epoch + 1) % 10 == 0 or epoch == 0:
            dist = val_action_counts / max(val_action_counts.sum(), 1) * 100
            dist_str = " | ".join(f"{action_names[j]}:{dist[j]:.0f}%" for j in range(n_actions))
            print(f"         Actions: {dist_str}")

    print(f"\nBest model at epoch {best_epoch+1} with val_reward={best_val_reward:.3f}")
    print(f"Saved to {model_path}")

    # Final analysis
    print("\n=== Final Analysis ===")
    policy.load(str(model_path))

    # Per-attack-type analysis
    attack_type_arr = data['attack_types']
    val_attack_types = attack_type_arr[val_idx]

    for atk_type in np.unique(val_attack_types):
        mask = val_attack_types == atk_type
        if mask.sum() == 0:
            continue

        atk_features = val_features[mask]
        atk_rewards = val_rewards[mask]

        # RL selector
        rl_total = 0.0
        rl_actions = np.zeros(n_actions, dtype=int)
        for i in range(len(atk_features)):
            action = policy.select_action(atk_features[i], greedy=True)
            rl_actions[action] += 1
            rl_total += atk_rewards[i, action]
        rl_avg = rl_total / len(atk_features)

        # Oracle
        oracle_avg = atk_rewards.max(axis=1).mean()

        # Best fixed
        fixed_avgs = atk_rewards.mean(axis=0)
        best_fixed = fixed_avgs.argmax()

        print(f"\n  {atk_type} ({mask.sum()} examples):")
        print(f"    RL selector:  {rl_avg:.3f}")
        print(f"    Oracle:       {oracle_avg:.3f}")
        print(f"    Best fixed:   {fixed_avgs[best_fixed]:.3f} ({action_names[best_fixed]})")
        dist = rl_actions / max(rl_actions.sum(), 1) * 100
        top_actions = sorted(range(n_actions), key=lambda j: -rl_actions[j])[:3]
        print(f"    Top actions:  " +
              ", ".join(f"{action_names[j]}:{dist[j]:.0f}%" for j in top_actions))


if __name__ == '__main__':
    main()
