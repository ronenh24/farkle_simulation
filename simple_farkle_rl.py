# Author: Ronen H
from architecture import SimpleActionReward
import torch
from utils import roll_loop, distance_scale
import random
import numpy as np
import optuna
from tqdm import trange
from strategy import SimpleRLStrategy, NaiveStrategy
from simulation import simulate_game
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def train_loop(action_reward: SimpleActionReward, action_opt: torch.optim.Optimizer, action_loss: torch.nn.Module,
               epsilon: float, gamma: float, step: int) -> tuple[SimpleActionReward, torch.optim.Optimizer]:
    """
    Trains simple RL deep Q-learning network over one game of two player Farkle.
    """
    player_1_score = 0
    player_2_score = 0
    turn = 1

    while max(player_1_score, player_2_score) < 10000:
        states = []
        rolls = []
        actual_rewards = []
        actions = []
        current_turn_score = 0
        num_dice = 6

        current_score = player_1_score
        advantage = player_1_score - player_2_score
        distance = 10000 - player_1_score
        if turn == 2:
            current_score = player_2_score
            advantage = player_2_score - player_1_score
            distance = 10000 - player_2_score

        while True:
            state = [distance, advantage, current_turn_score, num_dice, len(rolls)]
            dice_combination, roll_maxes, remaining_dice, farkled, legal_moves = roll_loop(num_dice)
            action = choose_action(farkled, num_dice, current_turn_score, current_score, roll_maxes, epsilon, step,
                                   legal_moves, action_reward, state)

            roll_score = roll_maxes[action - 1]
            actual_reward = compute_reward(gamma, len(rolls), roll_score, current_score, advantage, farkled)
            states.append(state)
            rolls.append([dice_combination, roll_score])
            actual_rewards.append(actual_reward)
            actions.append(action)
            current_turn_score += roll_score
            current_score += roll_score
            if farkled or action == 7:
                break
            distance -= roll_score
            advantage += roll_score
            num_dice = remaining_dice[action - 1]

        actual_rewards = np.cumsum(actual_rewards[::-1])[::-1].copy()
        if farkled:
            states = states[1:].copy()
            actions = actions[1:].copy()
            actual_rewards = actual_rewards[1:].copy()
        if len(states) > 0:
            action_reward.train()
            action_opt.zero_grad()
            predicted_rewards = action_reward.forward(torch.tensor(states).float()).\
                gather(1, torch.tensor(actions).unsqueeze(1) - 1)
            losses = action_loss.forward(predicted_rewards, torch.tensor(actual_rewards).unsqueeze(1).float())
            losses.backward()
            action_opt.step()

        if turn == 1:
            player_1_score = current_score
            turn = 2
        else:
            player_2_score = current_score
            turn = 1

    return action_reward, action_opt


def choose_action(farkled: bool, num_dice: int, current_turn_score: int, current_score: int, roll_maxes: list[int],
                  epsilon: float, step: int, legal_moves: list[int], action_reward: SimpleActionReward,
                  state: list[int]) -> int:
    if farkled:
        action = num_dice
        roll_maxes[action - 1] = -current_turn_score
    elif current_score + roll_maxes[7 - 1] >= 10000:
        action = 7
    elif random.uniform(0, 1) < epsilon ** step: # Exploration.
        action = random.choice(legal_moves)
    else: # Exploitation.
        action_reward.eval()
        with torch.no_grad():
            predicted_rewards = action_reward.forward(torch.tensor(state).float()). \
                gather(0, torch.tensor(legal_moves) - 1)
            probs = torch.softmax(predicted_rewards, 0).tolist()
            action = random.choices(legal_moves, probs)[0]
    return action


def compute_reward(gamma: float, num_rolls: int, roll_score: int, current_score: int,
                   advantage: int, farkled: bool) -> float:
    actual_reward = ((2 - gamma) ** np.log2(num_rolls + 1)) * roll_score # Increase per roll.
    distance_factor = 2 * distance_scale ** (current_score / 10000) / distance_scale # Increase as closer to 10,000.
    advantage_factor = 1
    if not farkled and advantage < 0:
        advantage_factor *= np.emath.logn(50, -advantage) # Increase as more behind.
    actual_reward *= distance_factor * advantage_factor
    return actual_reward


def train() -> None:
    """
    Trains the simple RL deep Q-learning network with optimal learning rate, epsilon (exploration), and gamma (reward).

    Saves the network weights as "simple_action_reward_state_dict.pt".

    Saves the turn score by current score for each side as "tables/training_simple_rl.csv" and
    "plots/training_simple_rl.csv"
    """
    study = optuna.create_study(study_name="Simple RL Optimization", direction="maximize")
    study.optimize(objective, 25)
    best_params = study.best_params
    print("Best Parameters:", best_params)

    lr = best_params["lr"]
    epsilon = best_params["epsilon"]
    gamma = best_params["gamma"]

    action_reward = SimpleActionReward()
    action_opt = torch.optim.AdamW(action_reward.parameters(), lr)
    action_loss = torch.nn.MSELoss()

    step = 0
    for _ in trange(0, 1000):
        action_reward, action_opt = train_loop(action_reward, action_opt, action_loss, epsilon, gamma, step)
        step += 0.01

    strategy = SimpleRLStrategy(action_reward, True)
    naive_strategy = NaiveStrategy(True)
    side_hist = []
    turns_hist = []
    scores_hist = []
    for _ in trange(0, 500):
        _, player_1_turns, _ = simulate_game(strategy, naive_strategy, True)
        turns_hist.extend(player_1_turns)
        scores_hist.append(0)
        scores_hist.extend(np.cumsum(player_1_turns)[:-1])
        side_hist.extend(["Player 1"] * len(player_1_turns))
    for _ in trange(0, 500):
        _, _, player_2_turns = simulate_game(naive_strategy, strategy, True)
        turns_hist.extend(player_2_turns)
        scores_hist.append(0)
        scores_hist.extend(np.cumsum(player_2_turns)[:-1])
        side_hist.extend(["Player 2"] * len(player_2_turns))

    torch.save(action_reward.state_dict(), "simple_action_reward_state_dict.pt")

    game_df = pd.DataFrame({"Side": side_hist, "Total Score": scores_hist, "Turn Score": turns_hist})
    game_df.to_csv("tables/training_simple_rl.csv", index=None)

    _, ax = plt.subplots(figsize=(20,20))
    sns.lineplot(game_df, x="Total Score", y="Turn Score", hue="Side", ax=ax)
    ax.grid()
    plt.title("Training Simple RL")
    plt.savefig("plots/training_simple_rl.jpg", dpi=500, bbox_inches="tight")
    plt.close()


def objective(trial: optuna.Trial) -> float:
    lr = trial.suggest_float("lr", 5e-6, 5e-3)
    epsilon = trial.suggest_float("epsilon", 0.75, 1)
    gamma = trial.suggest_float("gamma", 0.75, 1)

    action_reward = SimpleActionReward()
    action_opt = torch.optim.AdamW(action_reward.parameters(), lr)
    action_loss = torch.nn.MSELoss()

    step = 0
    for _ in trange(0, 1000):
        action_reward, action_opt = train_loop(action_reward, action_opt, action_loss, epsilon, gamma, step)
        step += 0.01

    strategy = SimpleRLStrategy(action_reward, True)
    naive_strategy = NaiveStrategy(True)
    wins_1 = 0
    for _ in trange(0, 500):
        wins_1 += simulate_game(strategy, naive_strategy, True)[0]
    wins_2 = 0
    for _ in trange(0, 500):
        wins_2 += int(simulate_game(naive_strategy, strategy, True)[0] == 0)
    print(wins_1, wins_2)
    print()

    return wins_1 + wins_2


def main():
    train()


if __name__ == "__main__":
    main()
