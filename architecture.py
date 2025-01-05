# Author: Ronen H
import torch


class SimpleActionReward(torch.nn.Module):
    """
    Input: State (distance, advantage, current turn score, number of dice, number of rolls).
    Output: Reward (keep 1, keep 2, keep 3, keep 4, keep 5, keep 6, stop).
    """
    def __init__(self):
        super(SimpleActionReward, self).__init__()
        self.reward = torch.nn.Sequential(
            torch.nn.Linear(5, 16),
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