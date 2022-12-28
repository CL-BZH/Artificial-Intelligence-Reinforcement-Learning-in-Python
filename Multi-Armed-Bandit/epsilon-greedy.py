import matplotlib.pyplot as plt
import numpy as np
#from typing import Callable
from typing import List
from typing import Optional
#from abc import ABC

from scipy import stats


NUM_TRIALS = 10000
EPSILON = 0.15
Bernoulli_p = [0.2, 0.5, 0.75]
Gaussian_params = [(1,1), (2,1), (3,1)]

class Epsilon:
    def __init__(self, epsilon: float):
        self.__epsilon = epsilon

    @property
    def epsilon(self):
        return self.__epsilon
    
    def __call__(self) -> float:
        """ The epsilon value is given by the derived class """
        raise NotImplementedError("epsilon() has to be implemented in the derived class")
        
class CstEpsilon(Epsilon):
    def __init__(self, epsilon: float):
        super().__init__(epsilon)

    def __call__(self) -> float:
        return self.epsilon
        
class ExpDecayEpsilon(Epsilon):
    """ Exponentialy decaying epsilon value """
    def __init__(self, epsilon: float):
        super().__init__(epsilon)
        self.__count = 0
        self.__alpha = 0.99

    def __call__(self) -> float:
        epsilon = self.epsilon * self.__alpha**self.__count
        self.__count += 1
        return epsilon
       
class LinDecayEpsilon(Epsilon):
    """ Linearly decaying epsilon value """
    def __init__(self, epsilon: float):
        super().__init__(epsilon)
        self.__count = 0
        self.__k = 0.0007

    def __call__(self) -> float:
        epsilon = max(self.epsilon - self.__k*self.__count, 0)
        print(epsilon)
        self.__count += 1
        return epsilon

           
class Distribution:
    
    def __init__(self, name: str = None):
        self.__name = name
        if self.__name == None:
            raise NotImplementedError("Distribution must have a name")
    
    @property
    def name(self):
        return self.__name


class Bandit:
    def __init__(self, R: Distribution):
        """ R is the distribution for the reward (Bernoulli or Gaussian) """
        self.R = R
        self.__N: int = 0 # Number of time this bandit was selected
        self.__e: float = 0 # Estimate of E
        # Define the type of a bandit as the distribution used for the reward
        self.type = R.__class__.__name__

    @property
    def N(self) -> int:
        return self.__N
    
    @property
    def e(self) -> float:
        return self.__e

    def expected_reward(self) -> float:
        return self.R.E
        
            
    def action(self) -> float:
        """ Take an action (pull the bandit's arm) and get a reward """
        # Since action is performed on the selected bandit increase by 1
        # the number of time the bandit was selected
        self.__N += 1
        reward = self.R()
        self.__e = ((self.__N - 1) * self.__e + reward) /  self.__N 
        return reward


class Bernoulli(Distribution):
    def __init__(self, p):
        """
        p: Bernoulli parameter
        """
        super().__init__("Bernoulli")
        self.p = p
        self.E = p

    def __call__(self) -> float:
        """ Returns 1 with probaility p (therefore 0 with probability 1-p) """
        return np.random.random() < self.p
        
class Gaussian(Distribution):
    def __init__(self, mean: float = 0., stdev: float = 1.):
        """
        mean: The mean of the Gaussian distribution
        stdev: The standard deviation of the Gaussian distribution
        """
        super().__init__("Gaussian")
        self.__mean = mean
        self.__stdev = stdev
        self.__tau = 1 / (self.__stdev**2)
        self.__E = mean
        self.__lambda_: float = 1.
    
    @property
    def E(self) -> float:
        return self.__mean
    
    @property
    def stdev(self) -> float:
        return self.__stdev
    
    @property
    def tau(self) -> float:
        return self.__tau
    
    @property
    def lambda_(self) -> float:
        return self.__lambda_
    
    def pdf(self, x: float):
        return stats.norm.pdf(x, self.E, self.stdev)
    
    def __add__(self, x ):
        self.__mean = (self.tau * x + self.lambda_ * self.E) / (self.tau + self.lambda_)
        self.__lambda_ += self.tau
        self.__stdev = 1 / np.sqrt(self.lambda_)
        self.__E = self.__mean
        return self
    
    def __call__(self) -> float:
        return np.random.normal(self.__mean, self.__stdev)

class Beta(Distribution):
    """
    See https://www.statlect.com/probability-distributions/beta-distribution
    """
    def __init__(self, alpha: float = 1., beta: float = 1.):
        super().__init__("Beta")
        self.__alpha = alpha
        self.__beta = beta
        self.__E: float = alpha / (alpha + beta)
        self.__stdev = np.sqrt((alpha * beta) / ((alpha + beta + 1)*((alpha+beta)**2)))
                               
    @property
    def alpha(self) -> float:
        return self.__alpha
    
    @alpha.setter
    def alpha(self, alpha: int) -> None:
        self.__alpha = alpha
    
    @property
    def beta(self) -> float:
        return self.__beta
    
    @beta.setter
    def beta(self, beta: int) -> None:
        self.__beta = beta

    @property
    def E(self) -> float:
        return self.__alpha / (self.__alpha + self.__beta)
    
    @property
    def stdev(self) -> float:
        return np.sqrt((self.alpha * self.beta) / ((self.alpha + self.beta + 1)*((self.alpha+self.beta)**2)))
    
    def pdf(self, x: float):
        return stats.beta.pdf(x, self.__alpha, self.__beta)
    
    def __add__(self, R ):
        self.__alpha += R
        self.__beta += 1 - R
        return self
    
    def __call__(self) -> float:
        return np.random.beta(self.alpha, self.beta)



class BanditAlgorithm:
    """ Interface for all algorithms that are used for the bandit selection. """
    
    def __init__(self, bandits: List[Bandit], epsilon: Epsilon = None):
        self.epsilon = epsilon
        self.bandits = np.array(bandits)
        self.bandits_count = len(bandits)
        self.bandits_indices = np.array(range(self.bandits_count))
        self.bandits_N: List[int] = np.zeros(self.bandits_count)
        self.chosen = None # Keep track of the last chosen bandit

    def explore(self) -> Optional[int]:
        if self.epsilon != None:
            if np.random.random() < self.epsilon():
                return np.random.randint(len(self.bandits))
        return None

    def exploit(self) -> int:
        """ The bandit selection for the explotation depends on the algorithm
        that is used (epsilon-greedy, UCB, Thompson-sampling,...) """
        raise NotImplementedError("exploit() has to be implemented in the derived class")

    def plot(self):
        raise NotImplementedError("plot() has to be implemented in the derived class")
    
    # We use the "Liskov substitution principle" for the update() method
    def update(self, *args, **kwargs) -> None:
        """ This method MUST BE CALLED once the action is taken """
        raise NotImplementedError("update() has to be implemented in the derived class")
        
    def __call__(self) -> Bandit:
         # Select the bandit using the explore/exploit method
        chosen = self.explore()
        if chosen is None:
            chosen = self.exploit()

        self.chosen = chosen
        
        return self.bandits[chosen]
    
    
class EpsilonGreedy(BanditAlgorithm):
    """ Implementation of the Epsilon-Greedy algorithm for bandit selection """

    def __init__(self, bandits: List[Bandit], epsilon: Epsilon = None) -> None:
        super().__init__(bandits, epsilon)
        self.bandits_Q: List[float] = np.zeros(self.bandits_count)
        
    def exploit(self) -> int:
        return np.argmax([bandit_Q for bandit_Q in self.bandits_Q])
   
    def update(self, reward, *args, **kwargs) -> None:
        # Update the bandit's Q
        cum_reward = self.bandits_N[self.chosen] * self.bandits_Q[self.chosen] + reward
        self.bandits_N[self.chosen] += 1
        self.bandits_Q[self.chosen] = cum_reward / self.bandits_N[self.chosen]

    def plot(self):
        pass

class OptimisticInitialValue(EpsilonGreedy):
    """ Implementation of the Optimistic Initial Values algorithm for bandit selection """
    
    def __init__(self, bandits: List[Bandit], initialVal: float = 5) -> None:
        super().__init__(bandits)
        self.bandits_Q = initialVal * np.ones(self.bandits_count)
        self.bandits_N = np.ones(self.bandits_count)
    
    def explore(self) -> Optional[int]:
        return None
    
    def plot(self):
        pass
    

class UCB(EpsilonGreedy):
    """ Implementation of the Upper-Confidence-Bound algorithm for bandit selection """
    def __init__(self, bandits: List[Bandit], epsilon: Epsilon, c: float = 2):
        self.c = c
        super().__init__(bandits, epsilon)
    
    def exploit(self) -> int:
        if any(self.bandits_N == 0):
            chosen = np.random.choice(self.bandits_indices)
        else:
            t = self.bandits_N.sum()
            uncertainty = np.sqrt(self.c * np.log(t) / self.bandits_N)
            chosen = np.argmax(self.bandits_Q + uncertainty)

        return chosen

    def plot(self):
        pass

class TS(BanditAlgorithm):

    """ 
    Thompson Sampling algorithm for bandit selection.
    The implementation is done only for a Bernoulli likelihood.
    """
    def __init__(self, bandits: List[Bandit], distribution: Distribution):
        super().__init__(bandits)
        self.__distribution = np.array([distribution() for _ in range(self.bandits_count)])
        if distribution == Beta:
            if any( [bandit.type != "Bernoulli" for bandit in self.bandits]):
                raise ValueError(f"Wrong conjugates {[distribution.name for distribution in self.__distribution]} for"
                                 f" bandits of type {[bandit.type for bandit in self.bandits]}")
        elif distribution == Gaussian:
            if any( [bandit.type != "Gaussian" for bandit in self.bandits]):
                raise ValueError(f"Wrong conjugates {[distribution.name for distribution in self.__distribution]} for"
                                 f" bandits of type {[bandit.type for bandit in self.bandits]}")
        else:
            raise ValueError(f"Wrong conjugates {[distribution.name for distribution in self.__distribution]} for"
                                 f" bandits of type {[bandit.type for bandit in self.bandits]}")
            
    @property
    def distribution(self):
        return self.__distribution
    
    def exploit(self) -> int:
        if any(self.bandits_N < 5):
            # Baysian stats suppose that we have a good enough prior
            chosen = np.random.choice(self.bandits_indices)
        else:
            theta_samples = [D() for D in self.distribution]
            chosen = np.argmax(theta_samples)
        return chosen

    def update(self, R) -> None:
        self.bandits_N[self.chosen] += 1
        self.distribution[self.chosen] += R

    def plot(self):
        argmin = np.argmin([D.E for D in self.distribution])
        x_min = self.__distribution[argmin].E - 3 * self.__distribution[argmin].stdev
        argmax = np.argmax([D.E for D in self.distribution])
        x_max = self.__distribution[argmax].E + 4 * self.__distribution[argmax].stdev
        x = np.linspace(x_min, x_max, 200)
        for i, D in enumerate(self.distribution):
            y = D.pdf(x)
            plt.plot(x, y, label=f"\u03B8: {self.bandits[i].expected_reward():.2f}, " 
                     #f"win rate = {D.alpha - 1}/{self.bandits[i].N} = "
                     #f"{(D.alpha - 1) / self.bandits[i].N: .3f}. "
                     f"\u03B8\u0302 = {D.E:.3f}")
        plt.title(f"Bandit distributions after {NUM_TRIALS} trials")
        plt.legend()
        plt.show()
        
class Test:
    def __init__(self, trials: int, algo: BanditAlgorithm):
        self.__trials = trials
        self.__algo = algo
        self.__gains = np.zeros(trials) # Gains at each step

    @property
    def trials(self) -> int:
        return self.__trials

    @property
    def gains(self) -> float:
        return self.__gains
    
    def run(self) -> None:
        # Run each trial
        for t in range(self.trials):
            # Select the bandit
            bandit: Bandit = self.__algo()
            
            # Action (pull the arm for the bandit)
            R = bandit.action()

            self.__algo.update(R)
            
            self.__gains[t] = R
    

def plot(test: Test, bandits: List[Bandit]):
    cumulative_gains = np.cumsum(test.gains)
    win_rates = cumulative_gains / (np.arange(test.trials) + 1)
    
    distribution = bandits[0].type
    max = np.max([bandit.expected_reward() for bandit in bandits])
    plt.title(f"Win rate evolution {distribution}")
    plt.xlabel("Number of action")
    plt.ylabel("Win rate")
    plt.plot(win_rates)
    plt.plot(np.ones(test.trials) * max)
    plt.show()


if __name__=="__main__":

    # List of Bandit object 
    # bandits = [Bandit(Bernoulli(p)) for p in Bernoulli_p]
    # or
    bandits = [Bandit(Gaussian(mean, stdev)) for (mean, stdev) in Gaussian_params]
    #bandits = [Bandit(Bernoulli(p)) for p in Bernoulli_p]

    # Epsilon for the epsilon-greedy algo
    # epsilon = CstEpsilon(EPSILON)
    epsilon = ExpDecayEpsilon(EPSILON)
    #epsilon = LinDecayEpsilon(EPSILON)
    
    # Algorithm for the bandit selection
    #algorithm = EpsilonGreedy(bandits, epsilon)
    # or (UCB with c=0.8 for example)
    #algorithm = UCB(bandits, epsilon, c=0.6)
    # or for Bernoulli Bandits
    #algorithm = TS(bandits, Beta)
    # or for Gaussian Bandits
    algorithm = TS(bandits, Gaussian)
    # or
    #algorithm = OptimisticInitialValue(bandits, 4)

    # Create the Test object and run
    test = Test(NUM_TRIALS, algorithm)
    test.run()

    # Plot
    plot(test, bandits)

    # Number of time each bandit was selected
    for bandit in bandits:
        print(f"Bandit with \u03B8 = {bandit.expected_reward()} "
              f"was selected {bandit.N} times. "
              f"\u03B8\u0302 = {bandit.e:.3f}")

    # In the case that Thompson Sampling is used, it is interesting
    # to see the distribution of the expected reward
    algorithm.plot()
    
