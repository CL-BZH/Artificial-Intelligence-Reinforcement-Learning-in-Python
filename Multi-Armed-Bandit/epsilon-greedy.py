import matplotlib.pyplot as plt
import numpy as np


NUM_TRIALS = 10000
EPSILON = 0.1
WIN_PROBABILITIES = [0.2, 0.5, 0.75]


class Bandit:
    def __init__(self, p):
        """
        p: The win rate (that is unknown in reality)
        p_hat: Estimated win rate
        N: Current number of sample collected
        """
        self.p = p
        self.p_hat = 0.
        self.N = 0.
        
    def pull(self):
        """ 
        Pull the bandit's arm and update its state.
        """
        # Draw a 1 (win) with probability p (hence 0 (loose) with proba 1-p)
        win = np.random.random() < self.p

        # Update the estimate probability of win
        sum = self.N * self.p_hat + win
        self.N += 1
        self.p_hat = sum / self.N

        return win



def run(trials, probabilities, epsilon):

    # Gains at each step
    gains = np.zeros(trials)
    
    # Create a Bandit object for each probability
    bandits = [Bandit(p) for p in probabilities]

    # Run each trial
    for t in range(trials):

        # Select the bandit using the explore/exploit method
        if np.random.random() < epsilon:
            # Explore
            bandit =  bandits[np.random.randint(len(bandits))]
        else:
            # Exploit
            bandit = bandits[np.argmax([bandit.p_hat for bandit in bandits])]

        # pull the arm for the bandit
        x = bandit.pull()

        gains[t] = x
        
    # Plotting
    cumulative_gains = np.cumsum(gains)
    win_rates =cumulative_gains / (np.arange(trials) + 1)
    plt.title("Win rate evolution")
    plt.xlabel("Number of pull")
    plt.ylabel("Win rate")
    plt.plot(win_rates)
    plt.show()

if __name__=="__main__":
    run(NUM_TRIALS, WIN_PROBABILITIES, EPSILON)
