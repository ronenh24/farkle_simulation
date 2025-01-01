# Author: Ronen H

import random


def strategy_one_two_three(current_score: int, goal_score: int, strategy: int, turn_score: int = 0, num_dice: int = 6) -> int:
    '''
    Uses Strategy 1, Strategy 2, or Strategy 3.

    current_score: Current score of player.
    goal_score: Winning score.
    strategy: Strategy player uses.
    turn_score: Turn score of player. Defaults to 0.
    num_dice: Number of dice to roll. Defaults to 6.

    Returns updated score of player after turn.
    '''
    turn = True
    while turn:
        roll_score = 0
        rolls = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
        for dice in range(num_dice):
            roll = random.randint(1, 6)
            rolls[roll] += 1
        roll_score, num_dice = _maximize_roll_score(rolls, num_dice)
        turn_score += roll_score

        if strategy == 1:
            turn = False
        elif roll_score == 0:
            turn = False
            turn_score = 0
        elif current_score + turn_score >= goal_score:
            turn = False
        elif num_dice == 0 and strategy == 2:
            num_dice = 6
        elif num_dice <= 2:
            turn = False
    
    return current_score + turn_score

def strategy_four_and_five(current_score: int, goal_score: int, strategy: int, turn_score: int = 0, num_dice: int = 6):
    '''
    Uses Strategy 4 or Strategy 5.

    current_score: Current score of player.
    goal_score: Winning score.
    strategy: Strategy player uses.
    turn_score: Turn score of player. Defaults to 0.
    num_dice: Number of dice to roll. Defaults to 6.

    Returns updated score of player after turn.
    '''
    turn = True
    while turn:
        rolls = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
        for dice in range(num_dice):
            roll = random.randint(1, 6)
            rolls[roll] += 1
        roll_score, num_dice = _maximize_single_score(rolls, num_dice)
        turn_score += roll_score
    
        if roll_score == 0:
            turn = False
            turn_score = 0
        elif current_score + turn_score >= goal_score:
            turn = False
        elif num_dice == 0 and strategy == 4:
            num_dice = 6
        elif num_dice <= 2:
            turn = False
    
    return current_score + turn_score

def strategy_six_and_seven(current_score: int, goal_score: int, deficit: int, strategy: int, turn_score: int = 0, num_dice: int = 6):
    '''
    Uses Strategy 6 or Strategy 7.

    current_score: Current score of player.
    goal_score: Winning score.
    deficit: How far behind player is from other player.
    strategy: Strategy player uses.
    turn_score: Turn score of player. Defaults to 0.
    num_dice: Number of dice to roll. Defaults to 6.

    Returns updated score of player after turn.
    '''
    turn = True
    while turn:
        rolls = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
        for dice in range(num_dice):
            roll = random.randint(1, 6)
            rolls[roll] += 1
        roll_score = None
        if strategy == 6:
            roll_score, num_dice = _maximize_roll_score(rolls, num_dice)
        else:
            roll_score, num_dice = _maximize_single_score(rolls, num_dice)
        turn_score += roll_score
        deficit -= roll_score

        if roll_score == 0:
            turn = False
            turn_score = 0
        elif deficit <= 0:
            turn = False
        elif num_dice == 0:
            num_dice = 6
    
    return current_score + turn_score
        

def strategy_eight_to_thirteen(current_score: int, goal_score: int, deficit: int, strategy: int, turn_score: int = 0, num_dice: int = 6):
    '''
    Uses Strategy 8, Strategy 9, Strategy 10, Strategy 11, Strategy 12, or Strategy 13.

    current_score: Current score of player.
    goal_score: Winning score.
    deficit: How far behind player is from other player.
    strategy: Strategy player uses.
    turn_score: Turn score of player. Defaults to 0.
    num_dice: Number of dice to roll. Defaults to 6.

    Returns updated score of player after turn.
    '''
    threshold = None

    # Player uses Strategy 8.
    if strategy == 8:
        threshold = 100

    # Player uses Strategy 9.
    elif strategy == 9:
        threshold = 200

    # Player uses Strategy 10.
    elif strategy == 10:
        threshold = 500
    
    # Player uses Strategy 11.
    elif strategy == 11:
        threshold = 1000
    
    # Player uses Strategy 12.
    elif strategy == 12:
        threshold = 1500
    
    # Player uses Strategy 13.
    else:
        threshold = 2000
    
    turn = True
    while turn and deficit > threshold:
        roll_score = 0
        rolls = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
        for dice in range(num_dice):
            roll = random.randint(1, 6)
            rolls[roll] += 1
        roll_score, num_dice = _maximize_roll_score(rolls, num_dice)
        turn_score += roll_score
        deficit -= roll_score

        if roll_score == 0:
            turn = False
            turn_score = 0
        elif deficit <= threshold:
            if current_score + turn_score >= goal_score:
                turn = False
            elif num_dice == 0:
                num_dice = 6
            elif num_dice <= 2:
                turn = False
        elif num_dice == 0:
            num_dice = 6

    if turn:
        return strategy_one_two_three(current_score, goal_score, 2, turn_score, num_dice)
    else:
        return current_score + turn_score

def _maximize_roll_score(rolls: dict[int, int], num_dice: int) -> tuple[int, int]:
    roll_score = 0
    while rolls[1] >= 3:
        roll_score += 1000
        rolls[1] -= 3
        num_dice -= 3
    while rolls[1] > 0:
        roll_score += 100
        rolls[1] -= 1
        num_dice -= 1
    while rolls[6] >= 3:
        roll_score += 600
        rolls[6] -= 3
        num_dice -= 3
    while rolls[5] >= 3:
        roll_score += 500
        rolls[5] -= 3
        num_dice -= 3
    while rolls[5] > 0:
        roll_score += 50
        rolls[5] -= 1
        num_dice -= 1
    while rolls[4] >= 3:
        roll_score += 400
        rolls[4] -= 3
        num_dice -= 3
    while rolls[3] >= 3:
        roll_score += 300
        rolls[3] -= 3
        num_dice -= 3
    while rolls[2] >= 3:
        roll_score += 200
        rolls[2] -= 3
        num_dice -= 3
    return roll_score, num_dice

def _maximize_single_score(rolls: dict[int, int], num_dice: int) -> tuple[int, int]:
    roll_score = 0
    if rolls[1] >= 3:
        roll_score += 1000
        num_dice -= 3
    elif rolls[6] >= 3:
        roll_score += 600
        num_dice -= 3
    elif rolls[5] >= 3:
        roll_score += 500
        num_dice -= 3
    elif rolls[4] >= 3:
        roll_score += 400
        num_dice -= 3
    elif rolls[3] >= 3:
        roll_score += 300
        num_dice -= 3
    elif rolls[2] >= 3:
        roll_score += 200
        num_dice -= 3
    elif rolls[1] > 0:
        roll_score += 100
        num_dice -= 1
    elif rolls[5] > 0:
        roll_score += 50
        num_dice -= 1
    return roll_score, num_dice

