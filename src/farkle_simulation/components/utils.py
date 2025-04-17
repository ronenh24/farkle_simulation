"""
Author: Ronen Huang
"""

from collections import Counter
from itertools import combinations
import random
import os
import orjson
from tqdm import tqdm


action_list = ["Keep 1", "Keep 2", "Keep 3", "Keep 4",
               "Keep 5", "Keep 6", "No Roll"]
max_score = {}
if os.path.isfile("max_score.jsonl"):
    with open("max_score.jsonl", "rb") as max_score_file:
        for max_score_line in tqdm(max_score_file):
            max_score_sub = orjson.loads(max_score_line)
            max_score[tuple(max_score_sub["dice_combination"])] =\
                max_score_sub["result"]


def score_combination(dice_combination: tuple) -> int:
    """
    Calculate score of dice combination.
    """
    count = Counter(dice_combination)
    score = 0
    dice_used = 0

    for die_value, cnt in count.items():
        while cnt >= 3:
            if die_value == 1:  # Three 1s.
                score += 1000
            else:  # Three of a kind.
                score += die_value * 100

            count[die_value] -= 3
            cnt -= 3
            dice_used += 3

    score += count[1] * 100  # One 1.
    score += count[5] * 50  # One 5.
    dice_used += count[1]
    dice_used += count[5]

    # Three pairs.
    if len(count) == 3 and all(v == 2 for v in count.values()):
        score = max(score, 750)
        if score == 750:
            dice_used = 6

    if len(count) == 6:  # One of each kind.
        score = max(score, 1000)
        if score == 1000:
            dice_used = 6

    if dice_used < len(dice_combination):
        score = 0

    return score


def max_scoring_combination(dice_combination: tuple) -> tuple[list[int],
                                                              list[int]]:
    """
    Max scoring combinations for 1 to 6 dices.
    """
    n = len(dice_combination)
    if dice_combination in max_score:
        return max_score[dice_combination][0].copy(), \
            max_score[dice_combination][1].copy()
    max_scores = []
    remaining_dice = []

    for i in range(1, n + 1):
        max_score_for_i = 0
        remaining_dice_for_i = 0
        for subset in combinations(dice_combination, i):
            score = score_combination(subset)
            if score > max_score_for_i:
                max_score_for_i = score
        if max_score_for_i > 0:
            remaining_dice_for_i = n - i
            if remaining_dice_for_i == 0:
                remaining_dice_for_i = 6

        max_scores.append(max_score_for_i)

        remaining_dice.append(remaining_dice_for_i)

    max_score[dice_combination] = (max_scores.copy(), remaining_dice.copy())

    return max_scores, remaining_dice


def roll_loop(num_dice: int) -> tuple[tuple, list[int], list[int],
                                      bool, list[int]]:
    """
    Roll specified number of dice with max scores, remaining dice,
    farkle status and legal moves.
    """
    dice_combination = tuple(
        sorted(random.choices(range(1, 6 + 1), k=num_dice))
    )
    max_scores, remaining_dice = max_scoring_combination(dice_combination)
    farkled = sum(max_scores) == 0
    max_scores.extend([0] * (7 - len(remaining_dice)))
    remaining_dice.extend([0] * (7 - len(remaining_dice)))
    max_scores[7 - 1] = max(max_scores)
    legal_moves = [i for i in range(1, num_dice + 1) if max_scores[i - 1] > 0]
    if not farkled:
        legal_moves.append(7)
    return dice_combination, max_scores, remaining_dice, farkled, legal_moves
