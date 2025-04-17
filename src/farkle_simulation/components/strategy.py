"""
Author: Ronen Huang
"""

import torch
import numpy as np
from farkle_simulation.components.architecture import SimpleActionReward
from farkle_simulation.components.utils import action_list, roll_loop


class BaseStrategy:
    """
    Base strategy which determines the action based on the state for a turn.
    """
    def __init__(self, train: bool = False):
        self.train = train

    def name(self) -> str:
        """
        Name of strategy.
        """
        pass

    def turn(self, current_score: int, advantage: int) -> int:
        """
        Turn score.
        """
        pass

    def action(self, state: list[int]) -> int:
        """
        Choose action.
        """
        pass


class SimpleRLStrategy(BaseStrategy):
    """
    Simple RL Strategy.
    """
    def __init__(self, action_reward: SimpleActionReward, train: bool = False):
        super().__init__(train)
        self.action_reward = action_reward
        self.action_reward.eval()

    def name(self) -> str:
        return "simple_rl"

    def turn(self, current_score: int, advantage: int) -> int:
        current_turn_score = 0
        num_dice = 6
        num_rolls = 0
        distance = 10000 - current_score
        with torch.no_grad():
            while True:
                dice_combination, roll_maxes, remaining_dice, farkled, \
                    legal_moves = roll_loop(num_dice)
                if not self.train:
                    print("Dice Combination -", dice_combination)
                if farkled:
                    if not self.train:
                        print("Farkled")
                    current_turn_score = 0
                    break
                action = self.action(
                    [
                        distance, advantage, current_turn_score, num_rolls
                    ] + roll_maxes, legal_moves
                )
                if not self.train:
                    print("Action -", action_list[action - 1])
                current_turn_score += roll_maxes[action - 1]
                print("Current Turn Score -", current_turn_score)
                print("Current Score -", current_score + current_turn_score)
                if action == 7:
                    break
                distance -= roll_maxes[action - 1]
                advantage += roll_maxes[action - 1]
                num_dice = remaining_dice[action - 1]
                num_rolls += 1
        return current_turn_score

    def action(self, state: list[int], legal_moves: list[int]) -> int:
        """
        Maximize reward based on state and legal moves.
        """
        predicted_rewards = self.action_reward.forward(
            torch.tensor(state).float()
        ).gather(0, torch.tensor(legal_moves) - 1)
        if not self.train:
            print("Predicted Rewards", predicted_rewards.tolist())
        action = legal_moves[predicted_rewards.argmax().item()]
        return action


class NaiveStrategy(BaseStrategy):
    """
    Naive Strategy.
    """
    def name(self) -> str:
        return "naive"

    def turn(self, current_score: int, advantage: int) -> int:
        current_turn_score = 0
        num_dice = 6
        distance = 10000 - current_score
        while True:
            dice_combination, roll_maxes, remaining_dice, farkled, _ =\
                roll_loop(num_dice)
            if not self.train:
                print("Dice Combination -", dice_combination)
            if farkled:
                if not self.train:
                    print("Farkled")
                current_turn_score = 0
                break
            action = self.action(
                [distance, advantage, remaining_dice], roll_maxes
            )
            if not self.train:
                print("Action -", action_list[action - 1])
            current_turn_score += roll_maxes[action - 1]
            print("Current Turn Score -", current_turn_score)
            print("Current Score -", current_score + current_turn_score)
            if action == 7:
                break
            distance -= roll_maxes[action - 1]
            advantage += roll_maxes[action - 1]
            num_dice = remaining_dice[action - 1]
        return current_turn_score

    def action(self, state: list[int], roll_maxes: list[int]) -> int:
        """
        Maximize roll score and stop if less than or equal to two dice left
        and disadvantage less than 1,000.
        """
        action = np.argmax(roll_maxes[:6]) + 1
        if state[0] - roll_maxes[7 - 1] <= 0 or\
            (state[1] > -1000 and state[2][action - 1] <= 2) or\
                (state[1] <= -1000 and state[2][action - 1] <= 1):
            action = 7
        return action


class CustomStrategy(BaseStrategy):
    """
    Manually chosen action.
    """
    def name(self) -> str:
        return "custom"

    def turn(self, current_score: int, advantage: int):
        current_turn_score = 0
        num_dice = 6
        while True:
            dice_combination, roll_maxes, remaining_dice, farkled, \
                legal_moves = roll_loop(num_dice)
            print("Dice Combination -", dice_combination)
            if farkled:
                print("Farkled")
                current_turn_score = 0
                break
            print(
                "Legal Moves -", ", ".join(
                    [
                        "(" + str(i) + ") " + str(action_list[i - 1]) +
                        " - max " + str(roll_maxes[i - 1]) +
                        " remaining dice " + str(remaining_dice[i - 1])
                        for i in legal_moves
                    ]
                )
            )
            set_legal_moves = set(legal_moves)
            while True:
                try:
                    action = int(input("Action - ").strip())
                    if action in set_legal_moves:
                        break
                except ValueError:
                    pass
            current_turn_score += roll_maxes[action - 1]
            print("Current Turn Score -", current_turn_score)
            print("Current Score -", current_score + current_turn_score)
            if action == 7:
                break
            num_dice = remaining_dice[action - 1]
        return current_turn_score
