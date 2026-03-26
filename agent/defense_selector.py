"""RL-based defense selector wrapping an OpenAttack classifier.

At inference time, the trained Q-network selects the best defense
per input text. Drop-in replacement for any static defense wrapper.

Requires BODEGA in PYTHONPATH for defense implementations.
"""

from typing import List

import numpy as np
import OpenAttack

from agent.features import TextFeatureExtractor
from agent.q_network import DefensePolicy
from agent.defense_env import DEFAULT_ACTION_SPACE, get_action_names
from defenses.preprocessing import get_defense


class RLDefenseSelector(OpenAttack.Classifier):
    """Adaptive defense selector using a trained Q-network."""

    def __init__(
        self,
        victim: OpenAttack.Classifier,
        policy_path: str,
        action_space=None,
        seed: int = 42,
        verbose: bool = False
    ):
        self.victim = victim
        self.action_space = action_space or DEFAULT_ACTION_SPACE
        self.verbose = verbose
        self.feature_extractor = TextFeatureExtractor()

        n_features = TextFeatureExtractor.NUM_FEATURES
        n_actions = len(self.action_space)
        action_names = get_action_names(self.action_space)

        self.policy = DefensePolicy(
            n_features=n_features,
            n_actions=n_actions,
            action_names=action_names
        )
        self.policy.load(policy_path)
        self.policy.q_net.eval()

        # Pre-create defense wrappers
        self.defenses = {}
        for i, (defense_name, param) in enumerate(self.action_space):
            if defense_name == 'none':
                self.defenses[i] = victim
            else:
                self.defenses[i] = get_defense(
                    defense_name, victim, param=param,
                    seed=seed, verbose=verbose
                )

        self.action_counts = np.zeros(n_actions, dtype=int)

    def get_pred(self, input_: List[str]) -> np.ndarray:
        return self.get_prob(input_).argmax(axis=1)

    def get_prob(self, input_: List[str]) -> np.ndarray:
        """For each text, select the best defense and apply it."""
        all_probs = []
        for text in input_:
            features = self.feature_extractor.extract(text)
            action = self.policy.select_action(features, greedy=True)
            self.action_counts[action] += 1

            if self.verbose:
                print(f"[RL] {self.policy.action_names[action]} | {text[:60]}...")

            prob = self.defenses[action].get_prob([text])
            all_probs.append(prob[0])

        return np.array(all_probs)

    def get_action_statistics(self) -> dict:
        """Return distribution of selected defenses."""
        total = max(self.action_counts.sum(), 1)
        stats = {}
        for i, name in enumerate(self.policy.action_names):
            count = int(self.action_counts[i])
            stats[name] = {'count': count, 'pct': round(count / total * 100, 1)}
        return stats

    def get_modifications(self):
        return []

    def save_modifications(self, path: str):
        pass

    def clear_modifications(self):
        pass

    def finalise(self):
        if hasattr(self.victim, 'finalise'):
            self.victim.finalise()
