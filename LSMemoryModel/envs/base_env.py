"""
Base Class for the Environments
"""
from tqdm import tqdm


class BaseEnv:
    def __init__(self, T):
        """
        Initializes the environment
        """
        self.name = "BaseEnv"
        self.T = T

    def update(self):
        """
        Updates the environments internal state
        """
        raise NotImplementedError

    def step(self, action):
        """
        Returns the reward for the action taken
        """
        raise NotImplementedError

    def reset(self):
        """
        Resets the environment
        """
        raise NotImplementedError

    def train(self):
        """
        Trains the algorithm
        """
        pbar = tqdm(total=self.T)
        for _ in range(self.T):
            action = self.algo.choose_action(self.context)
            r = self.step(action)
            self.algo.update(r)
            self.update(r)

            # print the reward
            pbar.set_description(f"Reward/time: {self.cum_rewards[-1]/self.t:.2f}")
            pbar.update(1)
