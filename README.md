# Author
Ronen H 

# Time Frame
November 2021 to December 2021 (modified for Python usage).  

# Farkle
The rules of Farkle can be seen in the *In a Nutshell* section of [https://farkle.games/official-rules/](https://farkle.games/official-rules/). In this case, there are two players with **Player 1** as the first player and **Player 2** as the second player.

The scoring system can be seen in the table below
<table style="margin-left: auto; margin-right: auto;">
  <tr><th>Dice to Keep</th> <th>Score</th>
  <tr><td>Three 1s</td> <td>1000</td>
  <tr><td>Three 6s</td> <td>600</td>
  <tr><td>Three 5s</td> <td>500</td>
  <tr><td>Three 4s</td> <td>400</td>
  <tr><td>Three 3s</td> <td>300</td>
  <tr><td>Three 2s</td> <td>200</td>
  <tr><td>One 1</td> <td>100</td>
  <tr><td>One 5</td> <td>50</td>
</table>
.  

# Strategies
The strategies that Player 1 and Player 2 can use are:
- Strategy 1: Roll once. Keep maximum roll score.
- Strategy 2: Keep rolling dice until two or less left. For each roll, keep maximum roll score. If there are no dice remaining, roll all six again.
- Strategy 3: Same as Strategy 2 but stop if no dice remaining.
- Strategy 4: Keep rolling dice until two or less left. For each roll, keep maximum single score. If there are no dice remaining, roll all sixe again.
- Strategy 5: Same as Strategy 4 but stop if no dice remaining.
- Strategy 6: Keep rolling dice until at least equal to the other score. For each roll, keep maximum roll score.
- Strategy 7: Same as Strategy 6 but keep maximum single score for each roll.
- Strategy 8: If behind by more than 100, for each roll, keep maximum roll score. Keep rolling dice until behind by 100 or less. Default to Strategy 2.
- Strategy 9: Same as Strategy 8 but by 200.
- Strategy 10: Same as Strategy 8 but by 500.
- Strategy 11: Same as Strategy 8 but by 1000.
- Strategy 12: Same as Strategy 8 but by 1500.
- Strategy 13: Same as Strategy 8 but by 2000.

.  

The implementation of the strategies is in `farkle_strategies.py`.  

# Simulation
The implementation of the simulation is in `farkle_simulation.py`.  

The `convergence_plot` function plots the expected probability Player 1 wins using some strategy with Player 2 using some other strategy.  

The `histogram` function plots the average probability Player 1 wins using some strategy with Player 2 using some other strategy.  

For example, if Player 1 uses Strategy 2 and Player 2 uses Strategy 8
```
from farkle_simulation import convergence_plot, histogram

# Expected probability Player 1 wins using Strategy 2 with Player 2 using Strategy 8.
convergence_plot(strategy_1=2, strategy_2=8)

# Average probabilities Player 1 wins using Strategy 2 with Player 2 using Strategy 8.
histogram(strategy_1=2, strategy_2=8)
```
.
