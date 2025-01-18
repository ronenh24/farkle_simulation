def main():
    from simulation import simulate_game
    from architecture import SimpleActionReward
    import torch
    from strategy import SimpleRLStrategy, NaiveStrategy

    action_reward = SimpleActionReward()
    action_reward.load_state_dict(torch.load("simple_action_reward_state_dict.pt", weights_only=True))

    simple_rl_strategy = SimpleRLStrategy(action_reward)
    naive_strategy = NaiveStrategy()

    simulate_game(naive_strategy, simple_rl_strategy)


if __name__ == "__main__":
    main()
