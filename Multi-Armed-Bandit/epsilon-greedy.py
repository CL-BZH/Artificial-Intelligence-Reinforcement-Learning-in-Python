import matplotlib.pyplot as plt
import numpy as np
from typing import Callable
from typing import List
from typing import Optional

NUM_TRIALS = 10000
EPSILON = 0.1
Bernoulli_p = [0.2, 0.5, 0.75]
Gaussian_params = [(5,2), (6,2), (1,5)]

class Bandit:
    def __init__(self, reward_fct):
        self.stat: float = 0.
        self.N: int = 0
        self.reward_fct = reward_fct
        # Define the type of a bandit as the distribution used for the reward
        self.bandit_type = reward_fct.__class__.__name__

    def pull(self):
        reward = self.reward_fct()

        # Update the estimate probability of win
        sum = self.N * self.stat + reward
        self.N += 1
        self.stat = sum / self.N

        return reward

class Bernoulli:
    def __init__(self, p):
        """
        p: Bernoulli parameter
        """
        self.p = p

    def __call__(self) -> float:
        """ Returns 1 with probaility p (therefore 0 with probability 1-p) """
        return np.random.random() < self.p
        
class Gaussian:
    def __init__(self, mean: float = 0., stdev: float = 1.):
        """
        mean: The mean of the Gaussian distribution
        stdev: The standard deviation
        """
        self.mean = mean
        self.stdev = stdev
        
    def __call__(self) -> float:
        return np.random.normal(self.mean, self.stdev)


class BanditAlgorithm:
    """ Interface for all algorithms that are used for the bandit selection. """
    
    def __init__(self, epsilon: float, bandits: List[Bandit]):
        self.epsilon = epsilon
        self.bandits = np.array(bandits)
        self.bandits_count = len(bandits)
        self.bandits_indices = np.array(range(self.bandits_count))
        self.N: List[int] = np.zeros(self.bandits_count)

    def explore(self) -> Optional[int]:
        if np.random.random() < self.epsilon:
            return np.random.randint(len(self.bandits))
        return None

    def exploit(self) -> int:
        raise NotImplementedError("exploit() has to be implemented in the derived class")
       
    def __call__(self) -> Bandit:
         # Select the bandit using the explore/exploit method
        chosen = self.explore()
        if chosen is None:
            chosen = self.exploit()

        self.N[chosen] += 1
        
        return self.bandits[chosen]

    
class EpsilonGreedy(BanditAlgorithm):
    """ Implementation of the Epsilon-Greedy algorithm for bandit selection """

    def exploit(self) -> int:
        return np.argmax([bandit.stat for bandit in self.bandits])

    
class UCB(BanditAlgorithm):
    """ Implementation of the Upper-Confidence-Bound algorithm for bandit selection """
    def __init__(self, epsilon: float, bandits: List[Bandit], c: float = 0.05):
        self.c = c
        super().__init__(epsilon, bandits)
    
    def exploit(self) -> int:
        if any(self.N == 0):
            chosen = np.random.choice(self.bandits_indices[self.bandits_indices == 0])
        else:
            t = self.N.sum()
            uncertainty = np.sqrt(np.log(t) / self.N)
            Q = np.array([bandit.stat for bandit in self.bandits])
            chosen = np.argmax(Q + self.c * uncertainty)

        return chosen


class Test:
    def __init__(self, trials: int, selection_fct):
        self.trials = trials
        self.selection_fct = selection_fct
        self.gains = np.zeros(trials) # Gains at each step
        
    def run(self):
        # Run each trial
        for t in range(self.trials):
            bandit: Bandit = self.selection_fct()
            
            # pull the arm for the bandit
            x = bandit.pull()

            self.gains[t] = x

        # Plotting
        cumulative_gains = np.cumsum(self.gains)
        win_rates =cumulative_gains / (np.arange(self.trials) + 1)
        
        distribution = bandit.bandit_type
        plt.title(f"Win rate evolution {distribution}")
        plt.xlabel("Number of pull")
        plt.ylabel("Win rate")
        plt.plot(win_rates)
        plt.show()
    


if __name__=="__main__":

    # Create a Bandit object for each probability p of reward
    bandits = [Bandit(Bernoulli(p)) for p in Bernoulli_p]

    selection = EpsilonGreedy(EPSILON, bandits)
    
    test = Test(NUM_TRIALS, selection)
    test.run()

    # Create a Bandit objects with Gaussian rewards
    bandits = [Bandit(Gaussian(mean, stdev)) for (mean, stdev) in Gaussian_params]

    # Use epsilon-greedy algo
    selection = EpsilonGreedy(EPSILON, bandits)
    
    test = Test(NUM_TRIALS, selection)
    test.run()

    # Use UCB algo
    selection = UCB(EPSILON, bandits, c=0.05)
    
    test = Test(NUM_TRIALS, selection)
    test.run()
