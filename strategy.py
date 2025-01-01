from architecture import SimpleActionReward
from utils import roll_loop
import torch
import numpy as np


class BaseStrategy:
    def __init__(self):
        pass

    def turn(self, current_score: int, advantage: int) -> int:
        pass

    def action(self, state: list[int], roll_maxes: list[int]) -> int:
        pass


class SimpleRLStrategy(BaseStrategy):
    def __init__(self, action_reward: SimpleActionReward):
        super(SimpleRLStrategy, self).__init__()
        self.action_reward = action_reward

    def turn(self, current_score: int, advantage: int) -> int:
        current_turn_score = 0
        num_dice = 6
        num_rolls = 0
        while True:
            _, roll_maxes, remaining_dice, farkled = roll_loop(num_dice)
            if farkled:
                current_turn_score = 0
                break
            action = self.action([advantage, current_turn_score, num_dice, num_rolls], roll_maxes)
            current_turn_score += roll_maxes[action - 1]
            if action == 7:
                current_turn_score += max(roll_maxes)
                break
            num_rolls += 1
        return current_turn_score

    def action(self, state: list[int], roll_maxes: list[int]) -> int:
        self.action_reward.eval()
        with torch.no_grad():
            predicted_rewards = self.action_reward(torch.tensor(state).float())
            max_predicted_reward = -np.inf
            for i in range(1, 7 + 1):
                if roll_maxes[i - 1] > 0 or (state[3] > 0 and i == 7):
                    if predicted_rewards[i - 1] > max_predicted_reward:
                        action = i
                        max_predicted_reward = predicted_rewards[i - 1]
        return action
