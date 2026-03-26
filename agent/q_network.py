"""Q-network for defense selection policy."""

import random
from typing import List

import numpy as np
import torch
import torch.nn as nn


class DefenseQNetwork(nn.Module):
    """Simple MLP mapping text features to Q-values for each defense action."""

    def __init__(self, n_features: int, n_actions: int, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_actions)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class DefensePolicy:
    """Epsilon-greedy policy over defense actions using Q-network."""

    def __init__(
        self,
        n_features: int,
        n_actions: int,
        action_names: List[str],
        lr: float = 1e-3,
        max_eps: float = 1.0,
        min_eps: float = 0.05,
        warmup_steps: int = 500,
        device: torch.device = None
    ):
        self.n_actions = n_actions
        self.action_names = action_names
        self.device = device or torch.device('cpu')

        self.q_net = DefenseQNetwork(n_features, n_actions).to(self.device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.eps = max_eps
        self.max_eps = max_eps
        self.min_eps = min_eps
        self.warmup_steps = warmup_steps
        self.step_count = 0

    def select_action(self, features: np.ndarray, greedy: bool = False) -> int:
        """Select defense action using epsilon-greedy policy."""
        if not greedy and random.random() < self.eps:
            return random.randint(0, self.n_actions - 1)

        with torch.no_grad():
            feat_t = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_values = self.q_net(feat_t)
            return q_values.argmax(dim=1).item()

    def get_q_values(self, features: np.ndarray) -> np.ndarray:
        """Get Q-values for all actions (for analysis)."""
        with torch.no_grad():
            feat_t = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
            return self.q_net(feat_t).cpu().numpy().flatten()

    def update(self, features_batch: np.ndarray, actions_batch: np.ndarray,
               rewards_batch: np.ndarray) -> float:
        """Update Q-network from a batch (contextual bandit, gamma=0)."""
        feat_t = torch.tensor(features_batch, dtype=torch.float32).to(self.device)
        act_t = torch.tensor(actions_batch, dtype=torch.long).to(self.device)
        rew_t = torch.tensor(rewards_batch, dtype=torch.float32).to(self.device)

        q_all = self.q_net(feat_t)
        q_selected = q_all.gather(1, act_t.unsqueeze(1)).squeeze(1)

        loss = self.loss_fn(q_selected, rew_t)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_count += 1
        if self.step_count < self.warmup_steps:
            self.eps = self.max_eps - (self.max_eps - self.min_eps) * self.step_count / self.warmup_steps
        else:
            self.eps = self.min_eps

        return loss.item()

    def save(self, path: str):
        torch.save({
            'q_net': self.q_net.state_dict(),
            'step_count': self.step_count,
            'eps': self.eps,
            'action_names': self.action_names,
            'n_features': self.q_net.network[0].in_features,
            'n_actions': self.q_net.network[-1].out_features,
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.q_net.load_state_dict(checkpoint['q_net'])
        self.step_count = checkpoint['step_count']
        self.eps = checkpoint['eps']
