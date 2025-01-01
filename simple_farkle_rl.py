import numpy as np

from utils import roll_loop, distance_scale, action_dict
import torch
import random
import optuna
from architecture import SimpleActionReward
from simulation import simulate_game
from strategy import SimpleRLStrategy


def train_loop(action_reward: SimpleActionReward, action_opt: torch.optim.Optimizer, action_loss: torch.nn.Module,
               gamma: float, step: int) -> tuple[SimpleActionReward, torch.optim.Optimizer, int,
                                                 list[tuple[int, float]]]:
    player_1_score = 0
    player_2_score = 0
    turn = 1
    losses = []

    while max(player_1_score, player_2_score) < 10000:
        print("Player 1 Score:", player_1_score)
        print("Player 2 Score:", player_2_score)
        print()
        states = []
        rolls = []
        actual_rewards = []
        actions = []
        current_turn_score = 0
        num_dice = 6

        advantage = player_1_score - player_2_score
        if turn == 2:
            advantage *= -1

        action_reward.eval()
        with torch.no_grad():
            while True:
                dice_combination, roll_maxes, remaining_dice, farkled = roll_loop(num_dice)
                if farkled:
                    action = num_dice
                    roll_maxes[action - 1] = -current_turn_score
                else:
                    if (turn == 1 and player_1_score + current_turn_score + max(roll_maxes) >= 10000) or \
                            (turn == 2 and player_2_score + current_turn_score + max(roll_maxes) >= 10000):
                        action = 7
                    else:
                        while True:
                            action = random.choice(range(1, 7 + 1))
                            if roll_maxes[action - 1] > 0 or (len(rolls) > 0 and action == 7):
                                break

                roll_score = roll_maxes[action - 1]
                actual_reward = (gamma ** len(rolls)) * roll_score
                distance_factor = (10000 - distance_scale * (player_1_score + current_turn_score)) / 10000 if turn == 1\
                    else (10000 - distance_scale * (player_2_score + current_turn_score)) / 10000
                distance_factor = 2 ** distance_factor / 2
                advantage_factor = 1
                if advantage > 0:
                    if farkled:
                        advantage_factor = np.log2(len(rolls) + 1)
                    else:
                        advantage_factor = 1 / (len(rolls) + 1)
                elif advantage < 0:
                    if farkled:
                        advantage_factor = 1 / 2
                    else:
                        advantage_factor = 2
                actual_reward *= distance_factor * advantage_factor
                if action == 7:
                    roll_score = max(roll_maxes)
                states.append([advantage, current_turn_score, num_dice, len(rolls)])
                rolls.append([dice_combination, roll_score])
                actual_rewards.append(actual_reward)
                actions.append(action)
                current_turn_score += roll_score
                advantage += roll_score

                if farkled or action == 7:
                    break
                num_dice = remaining_dice[action - 1]

        accumulated_actual = 0
        action_reward.train()
        for state, roll, actual_reward, action in zip(states[::-1], rolls[::-1], actual_rewards[::-1], actions[::-1]):
            accumulated_actual += actual_reward
            action_opt.zero_grad()
            predicted_reward = action_reward.forward(torch.tensor(state).float())[action - 1]
            loss = action_loss.forward(predicted_reward, torch.tensor(accumulated_actual).float())
            loss.backward()
            action_opt.step()
            losses.append((step, loss.item()))
            print("State:", state)
            print("Action:", action_dict[action])
            print("Roll:", roll)
            print("Predicted Reward:", predicted_reward.item())
            print("Actual Reward:", accumulated_actual)
            print()

        step += 1
        if turn == 1:
            player_1_score += current_turn_score
            turn = 2
        else:
            player_2_score += current_turn_score
            turn = 1
    print("Player 1 Score:", player_1_score)
    print("Player 2 Score:", player_2_score)
    print()

    return action_reward, action_opt, step, losses


def train():
    study = optuna.create_study(study_name="Simple RL Optimization", direction="maximize")
    study.optimize(objective, 1)


def objective(trial: optuna.Trial) -> float:
    lr = trial.suggest_float("lr", 1e-8, 1e-2)
    gamma = trial.suggest_float("gamma", 0.80, 0.99)

    action_reward = SimpleActionReward()
    action_opt = torch.optim.AdamW(action_reward.parameters(), lr)
    action_loss = torch.nn.MSELoss()

    step = 0
    for game in range(0, 1000):
        print("Game", game + 1)
        action_reward, action_opt, step, _ = train_loop(action_reward, action_opt, action_loss, gamma, step)

    wins = 0
    for game in range(0, 1000):
        print("Game", game + 1)
        wins += simulate_game(SimpleRLStrategy(action_reward), SimpleRLStrategy(action_reward))

    return wins


def main():
    train()


if __name__ == "__main__":
    main()
