"""Experience replay buffer for contextual bandit training."""

import random
from typing import Tuple

import numpy as np


class ReplayBuffer:
    """Fixed-size circular buffer storing (features, action, reward) tuples."""

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = []
        self.index = 0

    def push(self, features: np.ndarray, action: int, reward: float):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.index] = (features.copy(), action, reward)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        features = np.array([b[0] for b in batch])
        actions = np.array([b[1] for b in batch])
        rewards = np.array([b[2] for b in batch])
        return features, actions, rewards

    def __len__(self):
        return len(self.buffer)
