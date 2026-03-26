"""Defense selection environment -- evaluates defenses on adversarial texts.

Requires BODEGA in PYTHONPATH for defense implementations.
"""

from typing import List, Tuple

import numpy as np
import OpenAttack

from defenses.preprocessing import get_defense
from agent.features import TextFeatureExtractor


# Default action space: 8 representative defenses
DEFAULT_ACTION_SPACE = [
    ('none', 0.0),
    ('spellcheck', 0.0),
    ('unicode', 0.0),
    ('majority_vote', 3),
    ('majority_vote', 7),
    ('discretize', 0.0),
    ('spellcheck_mv', 3),
    ('char_noise', 0.10),
]

# Cost penalties for expensive defenses
DEFENSE_COSTS = {
    0: 0.0,    # none
    1: 0.0,    # spellcheck
    2: 0.0,    # unicode
    3: 0.05,   # majority_vote@3
    4: 0.10,   # majority_vote@7
    5: 0.0,    # discretize
    6: 0.05,   # spellcheck_mv@3
    7: 0.0,    # char_noise@0.10
}


def get_action_names(action_space: List[Tuple[str, float]] = None) -> List[str]:
    """Get human-readable names for each action."""
    action_space = action_space or DEFAULT_ACTION_SPACE
    names = []
    for name, param in action_space:
        if param == 0.0 or name == 'none':
            names.append(name)
        else:
            names.append(f"{name}@{param:g}")
    return names


class DefenseEnvironment:
    """Evaluates all defenses on a given text and returns rewards."""

    def __init__(
        self,
        victim: OpenAttack.Classifier,
        action_space: List[Tuple[str, float]] = None,
        seed: int = 42,
        verbose: bool = False
    ):
        self.victim = victim
        self.action_space = action_space or DEFAULT_ACTION_SPACE
        self.seed = seed
        self.verbose = verbose
        self.feature_extractor = TextFeatureExtractor()
        self.action_names = get_action_names(self.action_space)

        # Pre-create defense wrappers
        self.defenses = []
        for defense_name, param in self.action_space:
            if defense_name == 'none':
                self.defenses.append(victim)
            else:
                d = get_defense(defense_name, victim, param=param,
                                seed=seed, verbose=False)
                self.defenses.append(d)

    def evaluate_all_defenses(self, text: str, true_label: int) -> np.ndarray:
        """Evaluate all defenses on a text and return reward vector."""
        rewards = np.zeros(len(self.action_space), dtype=np.float32)
        for i, defense in enumerate(self.defenses):
            pred = defense.get_pred([text])[0]
            correct = 1.0 if pred == true_label else -1.0
            cost = DEFENSE_COSTS.get(i, 0.0)
            rewards[i] = correct - cost
        return rewards
