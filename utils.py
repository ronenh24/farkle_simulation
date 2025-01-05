import numpy as np
from collections import Counter
from itertools import combinations
import random


action_dict = {1: "Keep 1", 2: "Keep 2", 3: "Keep 3", 4: "Keep 4", 5: "Keep 5", 6: "Keep 6", 7: "No Roll"}
distance_scale = 2
dice_score = {}
max_dice_score = {}


# Farkle scoring function
def score_combination(dice_combination: tuple) -> int:
    if dice_combination in dice_score:
        return dice_score[dice_combination]
    count = Counter(dice_combination)
    score = 0
    dice_used = 0

    # Handle three of a kind and special cases for 1s
    for die_value, cnt in count.items():
        while cnt >= 3:
            if die_value == 1:
                score += 1000  # Three 1s earn 1000 points
            else:
                score += die_value * 100  # Three of a kind for other values

            # Subtract the three used dice from the count
            count[die_value] -= 3
            cnt -= 3
            dice_used -= 3

    # Add remaining 1s and 5s
    score += count[1] * 100  # Each 1 earns 100 points
    score += count[5] * 50  # Each 5 earns 50 points
    dice_used += count[1]
    dice_used += count[5]

    # Check for special combination: three pairs
    if len(count) == 3 and all(v == 2 for v in count.values()):
        score = max(score, 750)  # Three pairs earn 750 points
        if score == 750:
            dice_used = 6

    # Check for straight (1, 2, 3, 4, 5, 6)
    if dice_combination == (1, 2, 3, 4, 5, 6):
        score = max(score, 1000)  # A straight earns 1000 points
        if score == 1000:
            dice_used = 6

    if dice_used < len(dice_combination):
        score = 0

    dice_score[dice_combination] = score

    return score


# Function to calculate the maximum score for keeping up to n dice
def max_scoring_combination(dice_combination: list[int]) -> tuple[list[int], list[int]]:
    n = len(dice_combination)
    dice_combination = tuple(sorted(dice_combination))
    if dice_combination in max_dice_score:
        return max_dice_score[dice_combination]
    max_scores = []
    remaining_dice = []

    # Iterate over keeping up to 1, 2, ..., n dice
    for i in range(1, n + 1):  # Keep 1, 2, ..., n dice
        max_score_for_i = 0
        remaining_dice_for_i = 0
        # Generate all combinations of i dice from the full set of dice
        for subset in combinations(dice_combination, i):
            # Calculate the score for this subset of dice
            score = score_combination(subset)
            if score > max_score_for_i:
                max_score_for_i = score
        if max_score_for_i > 0:
            remaining_dice_for_i = n - i
            if remaining_dice_for_i == 0:
                remaining_dice_for_i = 6

        # Store the maximum score for keeping exactly i dice
        max_scores.append(max_score_for_i)

        remaining_dice.append(remaining_dice_for_i)

    max_dice_score[dice_combination] = (max_scores, remaining_dice)

    return max_scores, remaining_dice


def roll_loop(num_dice: int) -> tuple[list[int], list[int], list[int], bool, list[int]]:
    dice_combination = random.choices(range(1, 6 + 1), k=num_dice)
    max_scores, remaining_dice = max_scoring_combination(dice_combination)
    farkled = sum(max_scores) == 0
    max_scores.extend([0] * (7 - len(remaining_dice)))
    remaining_dice.extend([0] * (7 - len(remaining_dice)))
    max_scores[7 - 1] = max(max_scores)
    legal_moves = [i for i in range(1, num_dice + 1) if max_scores[i - 1] > 0]
    if not farkled:
        legal_moves.append(7)
    return dice_combination, max_scores, remaining_dice, farkled, legal_moves
