"""
Author: Ronen Huang
"""

import torch


class SimpleActionReward(torch.nn.Module):
    """
    Input: State (distance, advantage, current turn score, roll max keep 1,
                  roll max keep 2, roll max keep 3, roll max keep 4,
                  roll max keep 5, roll max keep 6, roll max stop).
    Output: Reward (keep 1, keep 2, keep 3, keep 4, keep 5, keep 6, stop).
    """
    def __init__(self):
        super(SimpleActionReward, self).__init__()
        self.reward = torch.nn.Sequential(
            torch.nn.Linear(11, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 7),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.reward(state)
