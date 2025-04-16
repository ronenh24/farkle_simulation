# Farkle Simulation

## Author
Ronen Huang

## Time Frame
November 2021 to December 2021 (modified for Python usage), January 2025 to Present (improvement).  

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
.  

## Strategies
The strategies that Player 1 and Player 2 can use are:
- Naive Strategy: For each roll, choose action that maximizes roll score and stop if number of dice is less than or equal to 2 and disadvantage of less than 1,000.
- Simple RL Strategy: For each roll, choose action that maximizes reward based on distance to 10,000, advantage, current turn score, number of dice, and number of rolls.

The implementation of the strategies is in `strategy.py`.

## Simple RL Strategy
Reward is based on roll score and increases as number of rolls does. If farkled, then the roll score is negative of the turn score.

The architecture of the deep Q-learning network which takes an input state (distance, advantage, current turn score, number of dice, number of rolls) and predicts reward for each action (take 1, take 2, take 3, take 4, take 5, take 6, stop) can be seen in `architecture.py`.

The training process can be seen in `simple_farkle_rl.py`. The plot of turn score by current score can be seen in **[plots/training_simple_rl.jpg](plots/training_simple_rl.jpg)** and the table in **[tables/training_simple_rl.csv](tables/training_simple_rl.csv)**.
![Training Simple RL](plots/training_simple_rl.jpg)

## Simulation
The implementation of the simulation is in `simulation.py`.  

The `convergence_plot` function plots the expected probability Player 1 wins using some strategy with Player 2 using some other strategy.  

The `histogram` function plots the average probability Player 1 wins using some strategy with Player 2 using some other strategy.  

For example, if player 1 uses the simple RL strategy and player 2 uses the naive strategy, the convergence plot can be seen in **[plots/convergence_simple_rl_naive.jpg](plots/convergence_simple_rl_naive.jpg)** and the true probability estimates can be seen in **[tables/convergence_simple_rl_naive.csv](tables/convergence_simple_rl_naive.csv)**.
![Convergence Plot Simple RL vs Naive](plots/convergence_simple_rl_naive.jpg)

# Pipeline
To train the simple RL agent and compare the strategies through convergence plot and histogram, run the below command on Powershell.
```
python pipeline.py
```
