# Farkle Simulation

## Author
Ronen Huang

## Time Frame
November 2021 to December 2021 (modified for Python usage), January 2025 to Present (add deep Q-Learning).  

## Install Farkle Simulation
The Farkle Simulation can be downloaded by pip.
```commandline
pip install farkle-simulation
```

## Farkle
The rules of Farkle can be seen in the *In a Nutshell* section of [https://farkle.games/official-rules/](https://farkle.games/official-rules/). In this case, there are two players with **Player 1** as the first player and **Player 2** as the second player.

The scoring system can be seen in the table below
<table style="margin-left: auto; margin-right: auto;">
  <tr><th>Dice to Keep</th> <th>Score</th>
  <tr><td>Three 1s or Straight</td> <td>1000</td>
  <tr><td>Three Different Pairs</td> <td>750</td>
  <tr><td>Three 6s</td> <td>600</td>
  <tr><td>Three 5s</td> <td>500</td>
  <tr><td>Three 4s</td> <td>400</td>
  <tr><td>Three 3s</td> <td>300</td>
  <tr><td>Three 2s</td> <td>200</td>
  <tr><td>One 1</td> <td>100</td>
  <tr><td>One 5</td> <td>50</td>
</table>

A player begins their turn with all six dice.
- If there are scoring combinations, a player can choose any and
  - Either keep rolling or stop
  - When number of dice is 0, it resets to 6
- Otherwise, a player has **"farkled"** and the turn score is **0**

Once a player reaches 10,000 points, they have won the game.

## Strategies
The implementation of the strategies is in [`strategy.py`](src/farkle_simulation/components/strategy.py).

The strategies that Player 1 and Player 2 can use are naive strategy, simple RL strategy, and custom strategy (manual). The `turn` function goes through one turn.
```python
from farkle_simulation.components.strategy import\
  NaiveStrategy, SimpleRLStrategy, CustomStrategy

naive_strategy = NaiveStrategy()
simple_rl_strategy = SimpleRLStrategy()
custom_strategy = CustomStrategy()

naive_turn_score = naive_strategy.turn(current_score, advantage)
simple_rl_turn_score = simple_rl_strategy.turn(current_score, advantage)
custom_turn_score = custom_strategy.turn(current_score, advantage)
```

### Custom Strategy
The player chooses the action each roll via typing 1 to 7 on the command line.

### Naive Strategy
For each roll
- Choose action that maximizes roll score
- Stop if number of dice is less than or equal to 2 and disadvantage of    less than 1,000.

### Simple Reinforcment Learning Strategy
For each roll
- Choose action that maximizes reward based on
  - Number of Rolls
    ```python
    # The gamma is reward factor decrease by roll.
    actual_reward = gamma ** np.log2(num_rolls + 1) * roll_score
    ```
  - Distance to 10,000 and Current Score
    ```python
    # Increases as close to 10,000.
    distance_factor = distance_scale ** (current_score / 10000)
    ```
  - Advantage
    ```python
    # Increase as more behind.
    advantage_factor = 1
    if advantage < 0:
        advantage_factor *= max(
          np.emath.logn(advantage_scale, -advantage), 1
        )
    ```
The architecture of the deep Q-learning networks takes 11 dimensional input state
- Distance
- Advantage
- Current Turn Score
- Roll Maxes

and predicts reward for each of 7 possible actions
- Keep 1 to 6 dice
- Stop

This can be seen in [`architecture.py`](src/farkle_simulation/components/architecture.py).

The training process can be seen at [`simple_farkle_rl.py`](src/farkle_simulation/components/simple_farkle_rl.py). The best model state dictionary is saved as [simple_action_reward_state_dict.pt](src/farkle_simulation/components/simple_action_reward_state_dict.pt).
```python
from farkle_simulation.components.simple_farkle_rl import train

train()
```

The plot of turn score by current score can be seen in **[training_simple_rl.jpg](src/plots/training_simple_rl.jpg)** and the table in **[training_simple_rl.csv](src/tables/training_simple_rl.csv)**.
![Training Simple RL](src/plots/training_simple_rl.jpg)

## Simulation
The implementation of the simulation is in [`simulation.py`](src/farkle_simulation/components/simulation.py).  

The `convergence_plot` function plots the expected probability Player 1 wins using some strategy with Player 2 using some other strategy.
```python
from farkle_simulation.components.simulation import convergence_plot

convergence_plot(naive_strategy, simple_rl_strategy)
convergence_plot(simple_rl_strategy, naive_strategy)
```

The convergence plots can be seen at **[convergence_naive_simple_rl.jpg](src/plots/convergence_naive_simple_rl.jpg)** ![Convergence Plot Naive vs Simple RL](src/plots/convergence_naive_simple_rl.jpg) and **[convergence_simple_rl_naive.jpg](src/plots/convergence_simple_rl_naive.jpg)** ![Convergence Plot Simple RL vs Naive](src/plots/convergence_simple_rl_naive.jpg)

The `histogram` function plots the average probability Player 1 wins using some strategy with Player 2 using some other strategy.
```python
from farkle_simulation.components.simulation import histogram

histogram(naive_strategy, simple_rl_strategy)
histogram(simple_rl_strategy, naive_strategy)
```

The histograms can be seen at **[histogram_naive_simple_rl.jpg](src/plots/histogram_naive_simple_rl.jpg)** ![Histogram Naive vs Simple RL](src/plots/histogram_naive_simple_rl.jpg) and **[histogram_simple_rl_naive.jpg](src/plots/histogram_simple_rl_naive.jpg)** ![Histogram Simple RL vs Naive](src/plots/histogram_simple_rl_naive.jpg)

## Pipeline
To train the simple RL agent and compare the strategies through convergence plot and histogram.
```python
from farkle_simulation.pipeline import train_simulate

train_simulate()
```

To play a game against the simple RL agent.
```python
from farkle_simulation.pipeline import play_game

play_game()
```
Input 1 to play as first player and 2 to play as second player.
```console
Player 1 (1) or Player (2)? 1
```
Input action to chose for dice combination.
```console
Dice Combination - (1, 1, 1, 1, 1, 4)
Legal Moves - (1) Keep 1 - max 100 remaining dice 5, (2) Keep 2 - max 200 remaining dice 4, (3) Keep 3 - max 1000 remaining dice 3, (4) Keep 4 - max 1100 remaining dice 2, (5) Keep 5 - max 1200 remaining dice 1, (7) No Roll - max 1200 remaining dice 0
Action - 7
```
The output of action is.
```console
Current Turn Score - 1200
Current Score - 1200
Turn Score - 1200
Player 1 Score - 1200 Player 2 Score - 0
```
The output of a "farkle" is.
```console
Dice Combination - (2, 6)
Farkled
Turn Score - 0
```
