"""
Author: Ronen Huang
"""

import os
import random
import torch
from torch.nn.functional import softmax, normalize
import numpy as np
import optuna
from tqdm import trange
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
from farkle_simulation.components.architecture import SimpleActionReward
from farkle_simulation.components.strategy import SimpleRLStrategy
from farkle_simulation.components.simulation import simulate_game
from farkle_simulation.components.utils import roll_loop


def train_loop(action_reward: SimpleActionReward,
               action_opt: torch.optim.Optimizer, action_loss: torch.nn.Module,
               distance_scale: float, advantage_scale: float,
               epsilon: float, gamma: float, step: int) ->\
        tuple[SimpleActionReward, torch.optim.Optimizer]:
    """
    Trains simple RL deep Q-learning network over one game of
    two player Farkle.
    """
    player_1_score = 0
    player_2_score = 0
    turn = 1

    while max(player_1_score, player_2_score) < 10000:
        states = []
        rolls = 0
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
            _, roll_maxes, remaining_dice, farkled, legal_moves =\
                roll_loop(num_dice)
            state = [distance, advantage, current_turn_score, rolls] +\
                roll_maxes
            action, roll_maxes = choose_action(
                farkled, num_dice, current_turn_score, current_score,
                roll_maxes, epsilon, step, legal_moves, action_reward, state
            )

            roll_score = roll_maxes[action - 1]
            actual_reward = compute_reward(
                distance_scale, advantage_scale, gamma,
                rolls, roll_score, current_score, advantage
            )
            states.append(state)
            actual_rewards.append(actual_reward)
            actions.append(action)
            current_turn_score += roll_score
            current_score += roll_score
            if farkled or action == 7:
                break
            rolls += 1
            distance -= roll_score
            advantage += roll_score
            num_dice = remaining_dice[action - 1]

        actual_rewards = np.cumsum(actual_rewards[::-1])[::-1].tolist()
        if farkled:
            states.pop()
            actions.pop()
            actual_rewards.pop()
        if len(states) > 0:
            action_reward.train()
            action_opt.zero_grad()
            predicted_rewards = action_reward.forward(
                torch.tensor(states).float()
            ).gather(1, torch.tensor(actions).unsqueeze(1) - 1)
            losses = action_loss.forward(
                predicted_rewards,
                torch.tensor(actual_rewards).unsqueeze(1).float()
            )
            losses.backward()
            action_opt.step()

        if turn == 1:
            player_1_score = current_score
            turn = 2
        else:
            player_2_score = current_score
            turn = 1

    return action_reward, action_opt


def choose_action(farkled: bool, num_dice: int, current_turn_score: int,
                  current_score: int, roll_maxes: list[int], epsilon: float,
                  step: int, legal_moves: list[int],
                  action_reward: SimpleActionReward, state: list[int]) ->\
                    tuple[int, list[int]]:
    """
    Choose action.
    """
    if farkled:
        action = num_dice
        roll_maxes[action - 1] = -current_turn_score
    elif current_score + roll_maxes[7 - 1] >= 10000:
        action = 7
    elif roll_maxes[num_dice - 1] > 0:
        action = num_dice
    # Exploration.
    elif random.uniform(0, 1) < epsilon ** step:
        action = random.choice(legal_moves)
    # Exploitation.
    else:
        action_reward.eval()
        with torch.no_grad():
            predicted_rewards = action_reward.forward(
                torch.tensor(state).float()
            ).gather(0, torch.tensor(legal_moves) - 1)
            probs = softmax(normalize(predicted_rewards, 1, 0), 0).tolist()
            action = random.choices(legal_moves, probs)[0]
    return action, roll_maxes


def compute_reward(distance_scale: float, advantage_scale: float,
                   gamma: float, num_rolls: int, roll_score: int,
                   current_score: int, advantage: int) -> float:
    """
    Compute reward.
    """
    if roll_score < 0:
        return (2 - gamma) ** np.log2(num_rolls + 1) * roll_score
    actual_reward = gamma ** np.log2(num_rolls + 1) * roll_score
    # Decrease as closer to 10,000.
    distance_factor = distance_scale ** (current_score / 10000)
    advantage_factor = 1
    # Increase as more behind.
    if advantage < 0:
        advantage_factor *= max(np.emath.logn(advantage_scale, -advantage), 1)
    actual_reward *= distance_factor * advantage_factor
    return actual_reward


def train() -> None:
    """
    Trains the simple RL deep Q-learning network with optimal learning rate,
    epsilon (exploration), and gamma (reward).

    Saves the network weights as "simple_action_reward_state_dict.pt".

    Saves the turn score by current score for each side as
    "tables/training_simple_rl.csv" and "plots/training_simple_rl.csv"
    """
    study = optuna.create_study(
        study_name="Simple RL Optimization", direction="maximize"
    )
    study.optimize(objective, 10)
    best_model_path =\
        "farkle_simulation/components/" +\
        "simple_action_reward_state_dict_" +\
        str(study.best_trial.number) + ".pt"
    action_reward = SimpleActionReward()
    action_reward.load_state_dict(
        torch.load(best_model_path, weights_only=True)
    )
    torch.save(
        action_reward.state_dict(), best_model_path
    )
    for i in range(0, 10):
        os.remove(
            "farkle_simulation/components/" +
            "simple_action_reward_state_dict_" +
            str(i) + ".pt"
        )

    strategy = SimpleRLStrategy(action_reward, True)
    side_hist = []
    turns_hist = []
    scores_hist = []
    for _ in trange(0, 5000):
        _, player_1_turns, player_2_turns =\
            simulate_game(strategy, strategy, True)
        turns_hist.extend(player_1_turns)
        scores_hist.append(0)
        scores_hist.extend(np.cumsum(player_1_turns)[:-1])
        side_hist.extend(["Player 1"] * len(player_1_turns))
        turns_hist.extend(player_2_turns)
        scores_hist.append(0)
        scores_hist.extend(np.cumsum(player_2_turns)[:-1])
        side_hist.extend(["Player 2"] * len(player_2_turns))

    game_df = pd.DataFrame(
        {
            "Side": side_hist,
            "Total Score": scores_hist,
            "Turn Score": turns_hist
        }
    )
    game_df.to_csv(
        "tables/training_simple_rl.csv", index=None
    )

    _, ax = plt.subplots(figsize=(20, 20))
    sns.lineplot(game_df, x="Total Score", y="Turn Score", hue="Side", ax=ax)
    ax.grid()
    plt.title("Training Simple RL")
    plt.savefig(
        "plots/training_simple_rl.jpg",
        dpi=500, bbox_inches="tight"
    )
    plt.close()


def objective(trial: optuna.Trial) -> float:
    """
    Return average turn score for player 1 and player 2.
    """
    lr = trial.suggest_float("lr", 5e-6, 5e-3)
    distance_scale = trial.suggest_float("distance_scale", 1, 5)
    advantage_scale = trial.suggest_int("advantage_scale", 50, 1000, step=50)
    epsilon = trial.suggest_float("epsilon", 0.75, 1)
    gamma = trial.suggest_float("gamma", 0.75, 1)
    step_increment = trial.suggest_float("step_increment", 0, 1)

    action_reward = SimpleActionReward()
    action_opt = torch.optim.AdamW(action_reward.parameters(), lr)
    action_loss = torch.nn.MSELoss()

    print()
    step = 0
    for _ in trange(0, 1000):
        action_reward, action_opt = train_loop(
            action_reward, action_opt, action_loss,
            distance_scale, advantage_scale, epsilon,
            gamma, step
        )
        step += step_increment

    model_path =\
        "farkle_simulation/components/" +\
        "simple_action_reward_state_dict_" + str(trial.number) + ".pt"
    torch.save(action_reward.state_dict(), model_path)

    strategy = SimpleRLStrategy(action_reward, True)
    turns_1 = []
    turns_2 = []
    for _ in trange(0, 5000):
        _, player_1_turns, player_2_turns = simulate_game(
            strategy, strategy, True
        )
        turns_1.extend(player_1_turns)
        turns_2.extend(player_2_turns)
    ci_lower_1 = st.t.interval(
        0.95, len(turns_1) - 1, np.mean(turns_1), st.sem(turns_1)
    )[0]
    ci_lower_2 = st.t.interval(
        0.95, len(turns_2) - 1, np.mean(turns_2), st.sem(turns_2)
    )[0]
    print(ci_lower_1, ci_lower_2)

    return (ci_lower_1 + ci_lower_2) / 2
