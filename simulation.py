from strategy import BaseStrategy


def simulate_game(player_1_strategy: BaseStrategy, player_2_strategy: BaseStrategy) -> int:
    player_1_score = 0
    player_2_score = 0
    turn = 1

    while max(player_1_score, player_2_score) < 10000:
        print("Player 1 Score:", player_1_score)
        print("Player 2 Score:", player_2_score)
        print()

        if turn == 1:
            player_1_score += player_1_strategy.turn(player_1_score, player_1_score - player_2_score)
            turn = 2
        else:
            player_2_score += player_2_strategy.turn(player_2_score, player_2_score - player_1_score)
            turn = 1

    print("Player 1 Score:", player_1_score)
    print("Player 2 Score:", player_2_score)
    print()

    return int(player_1_score > player_2_score)