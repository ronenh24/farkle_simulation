# Does not consider opponent score.

# Input: Advantage. Current turn score. Number of dice. Number of rolls.
# Output: Keep 1. Keep 2. Keep 3. Keep 4. Keep 5. Keep 6. Stay.
import torch


class SimpleActionReward(torch.nn.Module):
    def __init__(self):
        super(SimpleActionReward, self).__init__()
        self.reward = torch.nn.Sequential(
            torch.nn.Linear(4, 16),
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