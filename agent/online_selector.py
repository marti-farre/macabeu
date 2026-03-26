"""Online RL defense selector that learns during the attack.

Unlike the offline selector (frozen policy), this one updates its
Q-network after each attacked example, adapting in real-time to
the attacker's strategy.

Flow per example:
  1. Attacker queries victim multiple times to craft adversarial text
  2. Each query goes through our selector → picks defense based on features
  3. After the full attack on this example, we observe the outcome
  4. Update Q-network with this experience
  5. Next example benefits from the updated policy

Requires BODEGA in PYTHONPATH.
"""

from typing import List, Optional

import numpy as np
import torch
import OpenAttack

from agent.features import TextFeatureExtractor
from agent.q_network import DefensePolicy
from agent.replay_buffer import ReplayBuffer
from agent.defense_env import DEFAULT_ACTION_SPACE, get_action_names, DEFENSE_COSTS
from defenses.preprocessing import get_defense


class OnlineRLDefenseSelector(OpenAttack.Classifier):
    """Adaptive defense selector that learns online during attacks.

    Each call to get_prob/get_pred:
      - Extracts text features
      - Selects defense via epsilon-greedy
      - Tracks the (features, action) for later reward assignment

    Call `observe_result(true_label)` after each attacked example to
    inject the reward and update the Q-network.
    """

    def __init__(
        self,
        victim: OpenAttack.Classifier,
        action_space=None,
        seed: int = 42,
        lr: float = 1e-3,
        max_eps: float = 1.0,
        min_eps: float = 0.05,
        warmup_examples: int = 50,
        batch_size: int = 16,
        buffer_size: int = 5000,
        pretrained_path: Optional[str] = None,
        verbose: bool = False
    ):
        self.victim = victim
        self.action_space = action_space or DEFAULT_ACTION_SPACE
        self.verbose = verbose
        self.batch_size = batch_size
        self.feature_extractor = TextFeatureExtractor()

        n_features = TextFeatureExtractor.NUM_FEATURES
        n_actions = len(self.action_space)
        action_names = get_action_names(self.action_space)

        # Policy network
        self.policy = DefensePolicy(
            n_features=n_features,
            n_actions=n_actions,
            action_names=action_names,
            lr=lr,
            max_eps=max_eps,
            min_eps=min_eps,
            warmup_steps=warmup_examples
        )

        # Load pretrained weights if available (warm start)
        if pretrained_path:
            self.policy.load(pretrained_path)
            # Reset epsilon for online exploration
            self.policy.eps = max_eps
            self.policy.step_count = 0
            self.policy.warmup_steps = warmup_examples

        self.replay_buffer = ReplayBuffer(capacity=buffer_size)

        # Pre-create defense wrappers
        self.defenses = {}
        for i, (defense_name, param) in enumerate(self.action_space):
            if defense_name == 'none':
                self.defenses[i] = victim
            else:
                self.defenses[i] = get_defense(
                    defense_name, victim, param=param,
                    seed=seed, verbose=False
                )

        # Tracking for current example
        self._current_features = None
        self._current_action = None
        self._current_text = None

        # Statistics
        self.action_counts = np.zeros(n_actions, dtype=int)
        self.example_count = 0
        self.total_reward = 0.0
        self.reward_history = []

    def get_pred(self, input_: List[str]) -> np.ndarray:
        return self.get_prob(input_).argmax(axis=1)

    def get_prob(self, input_: List[str]) -> np.ndarray:
        """Select defense per text and apply it.

        Tracks the most recent (features, action) for reward assignment.
        """
        all_probs = []
        for text in input_:
            features = self.feature_extractor.extract(text)
            action = self.policy.select_action(features, greedy=False)
            self.action_counts[action] += 1

            # Track for reward assignment (keep last seen)
            self._current_features = features
            self._current_action = action
            self._current_text = text

            if self.verbose:
                print(f"[ONLINE_RL] eps={self.policy.eps:.3f} "
                      f"action={self.policy.action_names[action]} "
                      f"| {text[:50]}...")

            prob = self.defenses[action].get_prob([text])
            all_probs.append(prob[0])

        return np.array(all_probs)

    def observe_result(self, true_label: int, prediction_after_attack: int):
        """Called after each attacked example to provide reward.

        Args:
            true_label: Ground truth label for this example
            prediction_after_attack: What the defended victim predicted
                                     for the final adversarial text
        """
        if self._current_features is None:
            return

        # Reward: correct prediction = +1, wrong = -1, minus cost
        correct = 1.0 if prediction_after_attack == true_label else -1.0
        cost = DEFENSE_COSTS.get(self._current_action, 0.0)
        reward = correct - cost

        # Store experience
        self.replay_buffer.push(
            self._current_features,
            self._current_action,
            reward
        )

        self.total_reward += reward
        self.example_count += 1
        self.reward_history.append(reward)

        # Learn from replay buffer
        if len(self.replay_buffer) >= self.batch_size:
            f_batch, a_batch, r_batch = self.replay_buffer.sample(self.batch_size)
            loss = self.policy.update(f_batch, a_batch, r_batch)

            if self.verbose and self.example_count % 20 == 0:
                avg_recent = np.mean(self.reward_history[-20:])
                print(f"[ONLINE_RL] Example {self.example_count} | "
                      f"loss={loss:.4f} | eps={self.policy.eps:.3f} | "
                      f"avg_reward(20)={avg_recent:.3f}")

        # Reset tracking
        self._current_features = None
        self._current_action = None

    def get_action_statistics(self) -> dict:
        total = max(self.action_counts.sum(), 1)
        stats = {}
        for i, name in enumerate(self.policy.action_names):
            count = int(self.action_counts[i])
            stats[name] = {'count': count, 'pct': round(count / total * 100, 1)}
        return stats

    def get_learning_curve(self, window: int = 20) -> List[float]:
        """Return moving average of rewards over time."""
        if len(self.reward_history) < window:
            return self.reward_history
        curve = []
        for i in range(window, len(self.reward_history) + 1):
            curve.append(np.mean(self.reward_history[i - window:i]))
        return curve

    def get_modifications(self):
        return []

    def save_modifications(self, path: str):
        pass

    def clear_modifications(self):
        pass

    def finalise(self):
        if hasattr(self.victim, 'finalise'):
            self.victim.finalise()

    def save(self, path: str):
        self.policy.save(path)
