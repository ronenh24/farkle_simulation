# Author: Ronen H
from strategy import BaseStrategy
import matplotlib.pyplot as plt
from tqdm import trange
import seaborn as sns
import pandas as pd


def simulate_game(player_1_strategy: BaseStrategy, player_2_strategy: BaseStrategy,
                  train: bool=False) -> int | tuple[int, list[int], list[int]]:
    """
    Simulates one game of two player Farkle with the given strategies.
    """
    player_1_score = 0
    player_2_score = 0
    turn = 1
    if train:
        player_1_turns = []
        player_2_turns = []

    while max(player_1_score, player_2_score) < 10000:
        if not train:
            print("Player 1 Score:", player_1_score)
            print("Player 2 Score:", player_2_score)
            print()

        if turn == 1:
            player_1_turn = player_1_strategy.turn(player_1_score, player_1_score - player_2_score)
            player_1_score += player_1_turn
            if train:
                player_1_turns.append(player_1_turn)
            else:
                print("Roll Score", player_1_turn)
                print()
            turn = 2
        else:
            player_2_turn = player_2_strategy.turn(player_2_score, player_2_score - player_1_score)
            player_2_score += player_2_turn
            if train:
                player_2_turns.append(player_2_turn)
            else:
                print("Roll Score", player_2_turn)
                print()
            turn = 1

    if not train:
        print("Player 1 Score:", player_1_score)
        print("Player 2 Score:", player_2_score)

    win = int(player_1_score > player_2_score)
    if not train:
        print("Player 1 Wins") if win == 1 else print("Player 2 Wins")
        print()
        return win
    else:
        return win, player_1_turns, player_2_turns


def convergence_plot(player_1_strategy: BaseStrategy, player_2_strategy: BaseStrategy,
                     simulations: int=10, games: int=10000) -> None:
    """
    Convergence plot of estimated probability player 1 wins with given strategies.

    Saves estimates as "plots/convergence_[strategy 1 name]_[strategy 2 name].jpg" and
    "tables/convergence_[strategy 1 name]_[strategy 2 name].csv".
    """
    true_prob_estimates = []
    _, ax = plt.subplots(figsize=(20, 20))

    for simulation in range(0, simulations):
        print("Simulation", simulation + 1)

        total_wins = 0
        estimates = []
        for game in trange(0, games):
            total_wins += simulate_game(player_1_strategy, player_2_strategy, True)[0]
            if (game + 1) % 100 == 0:
                estimates.append(total_wins / (game + 1))
                print(estimates[-1])

        true_prob_estimates.append(total_wins / games)
        print(true_prob_estimates[-1])
        print()
        sns.lineplot(x=range(1, len(estimates) + 1), y=estimates, label="Simulation " + str(simulation + 1), ax=ax)

    ax.grid()
    plt.legend()
    plt.xlabel("Estimate")
    plt.ylabel("Estimated Probability Player 1 Wins")
    plt.title("Convergence Plot for Estimated Probability Player 1 Wins")
    plt.savefig("plots/convergence_" + player_1_strategy.name() + "_" + player_2_strategy.name() + ".jpg",
                dpi=500, bbox_inches="tight")
    plt.show()
    plt.close()

    true_prob_estimates_df = pd.DataFrame({"Simulation": range(1, simulations + 1),
                                           "Estimated Probability Player 1 Wins": true_prob_estimates})
    true_prob_estimates_df.to_csv("tables/convergence_" + player_1_strategy.name() + "_" +\
                                  player_2_strategy.name() + ".csv", index=None)


def histogram(player_1_strategy: BaseStrategy, player_2_strategy: BaseStrategy,
              simulations: int=500, games: int=1000) -> None:
    """
    Central Limit Theorem of probability player 1 wins with the given strategies.

    Saves histogram as "plots/histogram_[strategy 1 name]_[strategy 2 name].jpg" and
    "tables/histogram_[strategy 1 name]_[strategy 2 name].csv".
    """
    probs = []

    for simulation in range(0, simulations):
        print("Simulation", simulation + 1)

        total_wins = 0
        for _ in trange(0, games):
            total_wins += simulate_game(player_1_strategy, player_2_strategy, True)[0]

        probs.append(total_wins / games)
        print(probs[-1])
        print()

    _, ax = plt.subplots(figsize=(20, 20))
    sns.histplot(x=probs, bins=simulations // 4, ax=ax)
    ax.grid()
    plt.xlabel("Probability Player 1 Wins")
    plt.ylabel("Count")
    plt.title("Histogram for Probability Player 1 Wins")
    plt.savefig("plots/histogram_" + player_1_strategy.name() + "_" + player_2_strategy.name() + ".jpg",
                dpi=500, bbox_inches="tight")
    plt.show()
    plt.close()

    probs_df = pd.DataFrame({"Simulation": range(1, simulations + 1), "Probability Player 1 Wins": probs})
    probs_df.to_csv("tables/histogram_" + player_1_strategy.name() + "_" + player_2_strategy.name() + ".csv",
                    index=None)


def main():
    from architecture import SimpleActionReward
    import torch
    from strategy import SimpleRLStrategy, NaiveStrategy

    action_reward = SimpleActionReward()
    action_reward.load_state_dict(torch.load("simple_action_reward_state_dict.pt", weights_only=True))

    simple_rl_strategy = SimpleRLStrategy(action_reward, True)
    naive_strategy = NaiveStrategy(True)

    convergence_plot(simple_rl_strategy, naive_strategy, 10, 10000)
    convergence_plot(naive_strategy, simple_rl_strategy, 10, 10000)

    histogram(simple_rl_strategy, naive_strategy, 500, 1000)
    histogram(naive_strategy, simple_rl_strategy, 500, 1000)


if __name__ == "__main__":
    main()
