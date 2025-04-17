"""
Author: Ronen Huang
"""

from importlib.resources import files
import torch
import orjson
from farkle_simulation.components.simple_farkle_rl import train
from farkle_simulation.components.architecture import SimpleActionReward
from farkle_simulation.components.strategy import SimpleRLStrategy, NaiveStrategy, CustomStrategy
from farkle_simulation.components.simulation import convergence_plot, histogram
from farkle_simulation.components.utils import max_score


def play_game():
    """
    Plays a game of Farkle against simple RL strategy.
    """
    # Specify whether player is first or second.
    while True:
        try:
            player_num = int(input("Player 1 (1) or Player (2)? ").strip())
            if player_num == 1 or player_num == 2:
                break
        except ValueError:
            pass

    turn = 1
    player_1_score = 0
    player_2_score = 0

    best_model_file = "simple_action_reward_state_dict.pt"
    best_model_path = files("farkle_simulation.components") / best_model_file
    if not best_model_path.is_file:  # Train Simple RL if no model.
        train()

    action_reward = SimpleActionReward()
    action_reward.load_state_dict(
        torch.load(best_model_path, weights_only=True)
    )

    if player_num == 1:
        player_1_strategy = CustomStrategy()
        player_2_strategy = SimpleRLStrategy(action_reward)
    else:
        player_2_strategy = CustomStrategy()
        player_1_strategy = SimpleRLStrategy(action_reward)

    while player_1_score < 10000 and player_2_score < 10000:
        if turn == 1:
            turn_score = player_1_strategy.turn(
                player_1_score, player_1_score - player_2_score
            )
            player_1_score += turn_score
            turn = 2
        else:
            turn_score = player_2_strategy.turn(
                player_2_score, player_2_score - player_1_score
            )
            player_2_score += turn_score
            turn = 1
        print("Turn Score -", turn_score)
        print(
            "Player 1 Score -", player_1_score,
            "Player 2 Score -", player_2_score
        )
        print()


def train_simulate():
    """
    Train simple RL and evaluate effectiveness.
    """
    train()

    naive_strategy = NaiveStrategy()

    best_model_file = "simple_action_reward_state_dict.pt"
    best_model_path = files("farkle_simulation.components") / best_model_file
    action_reward = SimpleActionReward()
    action_reward.load_state_dict(
        torch.load(best_model_path, weights_only=True)
    )
    simple_rl_strategy = SimpleRLStrategy(action_reward)

    convergence_plot(naive_strategy, simple_rl_strategy)
    convergence_plot(simple_rl_strategy, naive_strategy)

    histogram(naive_strategy, simple_rl_strategy)
    histogram(simple_rl_strategy, naive_strategy)

    with open(
        "farkle_simulation/components/max_score.jsonl", "wb"
    ) as max_score_file:
        for dice_combination, result in max_score.items():
            max_score_file.write(
                orjson.dumps(
                    {
                        "dice_combination": dice_combination,
                        "result": result
                    }
                ) + b"\n"
            )
