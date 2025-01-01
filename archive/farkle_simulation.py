# Author: Ronen H

from tqdm import tqdm
from farkle_strategies import strategy_one_two_three, strategy_four_and_five, strategy_six_and_seven, strategy_eight_to_thirteen
import matplotlib.pyplot as plt


def convergence_plot(strategy_1: int, strategy_2: int, simulations: int = 10, games: int = 100000) -> None:
    '''
    Convergence plot of the estimated probability of winning for Player 1.

    strategy_1: Strategy Player 1 uses (1 to 13).
    strategy_2: Strategy Player 2 uses (1 to 13).
    simulations: Number of simulations to run. Defaults to 10.
    games: Number of Farkle games per simulation. Defaults to 100,000.
    '''
    # List of the estimates of
    # the true probability Player 1 
    # wins.
    true_prob_estimates = []

    # Simulate specified runs of specified games of farkle.
    for simulation in tqdm(range(simulations)):
        
        # Number of times Player 1 wins 
        # during a run.
        total_wins_1 = 0
        
        # Winning score.
        goal_score = 10000
        
        # List of 100 estimates of the true 
        # probability Player 1 wins for the run. 
        # An estimate is taken every 1000 games.
        estimates = []
        
        # The list of the number of estimates 
        # from 1 to 100. Used as the x-axis
        # for the convergence plot.
        estimate_num = []
        for num in range(100):
            estimate_num.append(num + 1)
            
        # Simulates games of farkle.
        for game in range(games):
            
            # Current score of Player 1 for the game.
            current_score_1 = 0
            
            # Current score of Player 2 for the game.
            current_score_2 = 0
            
            # Turn is 1 if it is Player 1's turn. Turn
            # is 2 if it is Player 2's turn. It is Player 1's
            # turn first.
            turn = 1
            
            # Simulate a game of farkle. The game ends when 
            # one of the player's scores is equal to or greater 
            # than the 10000.
            while current_score_1 < goal_score and current_score_2 < goal_score:
                
                # Player 1's turn.
                if turn == 1:

                    # Player 1 uses Strategy 1, Strategy 2, or Strategy 3.
                    if strategy_1 == 1 or strategy_1 == 2 or strategy_1 == 3:
                        current_score_1 = strategy_one_two_three(current_score_1, goal_score, strategy_1)
                    
                    # Player 1 uses Strategy 4 or Strategy 4.
                    elif strategy_1 == 4 or strategy_1 == 5:
                        current_score_1 = strategy_four_and_five(current_score_1, goal_score, strategy_1)

                    else:
                        deficit = current_score_2 - current_score_1

                        # Player 1 uses Strategy 6 or Strategy 7.
                        if strategy_1 == 6 or strategy_1 == 7:
                            current_score_1 = strategy_six_and_seven(current_score_1, goal_score, deficit, strategy_1)
                        
                        # Player 1 uses Strategy 8, Strategy 9, Strategy 10, Strategy 11, Strategy 12, or Strategy 13.
                        else:
                            current_score_1 = strategy_eight_to_thirteen(current_score_1, goal_score, deficit, strategy_1)
                    
                    turn = 2

                # Player 2's turn.
                else:
                    
                    # Player 2 uses Strategy 1, Strategy 2, or Strategy 3.
                    if strategy_2 == 1 or strategy_2 == 2 or strategy_2 == 3:
                        current_score_2 = strategy_one_two_three(current_score_2, goal_score, strategy_2)
                    
                    # Player 2 uses Strategy 4 or Strategy 4.
                    elif strategy_2 == 4 or strategy_2 == 5:
                        current_score_2 = strategy_four_and_five(current_score_2, goal_score, strategy_2)
                    
                    else:
                        deficit = current_score_1 - current_score_2

                        # Player 2 uses Strategy 6 or Strategy 7.
                        if strategy_2 == 6 or strategy_2 == 7:
                            current_score_2 = strategy_six_and_seven(current_score_2, goal_score, deficit, strategy_2)
                        
                        # Player 2 uses Strategy 8, Strategy 9, Strategy 10, Strategy 11, Strategy 12, or Strategy 13.
                        else:
                            current_score_2 = strategy_eight_to_thirteen(current_score_2, goal_score, deficit, strategy_2)
                        
                    turn = 1
            
            # Adds to the total number of wins 
            # Player 1 has on the current run if 
            # the game ends with Player 1 having 
            # a score equal to or greater than the 
            # 10000.
            if current_score_1 >= goal_score:
                total_wins_1 += 1
            
            # For every 1000 games played, add estimate of 
            # true probability Player 1 wins to the list of 
            # estimates based off total wins so far and the 
            # number of games played so far.
            if (game + 1) % 1000 == 0:
                estimates.append(1.0 * total_wins_1 / (game + 1))
            
            # If all games has beeen played, add estimate of 
            # true probability Player 1 wins to the list of 
            # estimates of the true probability Player 1 
            # wins.
            if game + 1 == games:
                true_prob_estimates.append(1.0 * total_wins_1 / games)
        # Plot the number of estimates and the 
        # estimates for the run to the convergence plot.
        plt.plot(estimate_num, estimates)

    # Saves the convergence plot as a .png file. The name  
    # of the file is changed according to the different 
    # strategies used. For instance, if Player 1 
    # uses Strategy 2 and Player 2 uses Strategy 8,
    # the file name is convergence_2_8.png
    plt.ylabel('Estimated Probability of Winning For Player 1')
    plt.savefig('plots/convergence_' + str(strategy_1) + '_' + str(strategy_2) + '.png',
                dpi = 500, bbox_inches = 'tight')
    
    # Prints the estimates of the true probability 
    # Player 1 wins.
    print(f'The estimates of the true probability Player 1 wins are {true_prob_estimates}.')

    # Prints the largest percentile Confidence Interval for the 
    # true probability Player 1 wins which is between 
    # the lowest estimate and highest estimate.
    min_estimate = min(true_prob_estimates)
    max_estimate = max(true_prob_estimates)
    print(f'The largest percentile Confidence Interval for the true probability Player 1 wins is [{min_estimate}, {max_estimate}].')

def histogram(strategy_1: int, strategy_2: int, simulations: int = 500, games: int = 1000) -> None:
    '''
    Histogram of the average probability of winning for Player 1.

    strategy_1: Strategy Player 1 uses (2, 8, 9, 10, 11, 12, or 13).
    strategy_2: Strategy Player 2 uses (2, 8, 9, 10, 11, 12, or 13).
    simulations: Number of simulations to run. Defaults to 500.
    games: Number of Farkle games per simulation. Defaults to 1,000.
    '''
    # List of average probabilities that 
    # Player 1 wins from all runs.
    winning_probs = []

    # Simulate specified runs of specified games of farkle.
    for simulation in tqdm(range(simulations)):
        
        # Number of times Player 1 wins 
        # during a run.
        total_wins_1 = 0
        
        # Winning score.
        goal_score = 10000
            
        # Simulates games of farkle.
        for game in range(games):
            
            # Current score of Player 1 for the game.
            current_score_1 = 0
            
            # Current score of Player 2 for the game.
            current_score_2 = 0
            
            # Turn is 1 if it is Player 1's turn. Turn
            # is 2 if it is Player 2's turn. It is Player 1's
            # turn first.
            turn = 1
            
            # Simulate a game of farkle. The game ends when 
            # one of the player's scores is equal to or greater 
            # than the 10000.
            while current_score_1 < goal_score and current_score_2 < goal_score:
                
                # Player 1's turn.
                if turn == 1:

                    # Player 1 uses Strategy 1, Strategy 2, or Strategy 3.
                    if strategy_1 == 1 or strategy_1 == 2 or strategy_1 == 3:
                        current_score_1 = strategy_one_two_three(current_score_1, goal_score, strategy_1)
                    
                    # Player 1 uses Strategy 4 or Strategy 4.
                    elif strategy_1 == 4 or strategy_1 == 5:
                        current_score_1 = strategy_four_and_five(current_score_1, goal_score, strategy_1)

                    else:
                        deficit = current_score_2 - current_score_1

                        # Player 1 uses Strategy 6 or Strategy 7.
                        if strategy_1 == 6 or strategy_1 == 7:
                            current_score_1 = strategy_six_and_seven(current_score_1, goal_score, deficit, strategy_1)
                        
                        # Player 1 uses Strategy 8, Strategy 9, Strategy 10, Strategy 11, Strategy 12, or Strategy 13.
                        else:
                            current_score_1 = strategy_eight_to_thirteen(current_score_1, goal_score, deficit, strategy_1)
                    
                    turn = 2

                # Player 2's turn.
                else:
                    
                    # Player 2 uses Strategy 1, Strategy 2, or Strategy 3.
                    if strategy_2 == 1 or strategy_2 == 2 or strategy_2 == 3:
                        current_score_2 = strategy_one_two_three(current_score_2, goal_score, strategy_2)
                    
                    # Player 2 uses Strategy 4 or Strategy 4.
                    elif strategy_2 == 4 or strategy_2 == 5:
                        current_score_2 = strategy_four_and_five(current_score_2, goal_score, strategy_2)
                    
                    else:
                        deficit = current_score_1 - current_score_2

                        # Player 2 uses Strategy 6 or Strategy 7.
                        if strategy_2 == 6 or strategy_2 == 7:
                            current_score_2 = strategy_six_and_seven(current_score_2, goal_score, deficit, strategy_2)
                        
                        # Player 2 uses Strategy 8, Strategy 9, Strategy 10, Strategy 11, Strategy 12, or Strategy 13.
                        else:
                            current_score_2 = strategy_eight_to_thirteen(current_score_2, goal_score, deficit, strategy_2)
                        
                    turn = 1
            
            # Adds to the total number of wins 
            # Player 1 has on the current run if 
            # the game ends with Player 1 having 
            # a score equal to or greater than the 
            # 10000.
            if current_score_1 >= goal_score:
                total_wins_1 += 1
            
        # The average probability of Player 1 winning 
        # for the current run is added to the list of 
        # average probabilities of Player 1 winning.
        # winning_prob = 1.0 * total_wins_1 / games
        # winning_probs.append(winning_prob)

    # Plots the histogram of average probabilities
    # Player 1 wins. 
    plt.hist(winning_probs, bins = 125)
    plt.xlabel('Probability')
    plt.ylabel('Count')

    # Saves the histogram as a .png file. The name of the 
    # file is changed according to the different 
    # strategies used. For instance, if Player 1 
    # uses Strategy 2 and Player 2 uses Strategy 8,
    # the file name should be changed to 
    # hist_2_8.png
    plt.savefig('plots/hist_' + str(strategy_1) + '_' + str(strategy_2) + '.png',
                dpi = 500, bbox_inches = 'tight')


if __name__ == '__main__':
    # Can set Strategy of Player 1 and Strategy of Player 2.
    convergence_plot(2, 13)
    plt.clf()
    histogram(2, 13)

