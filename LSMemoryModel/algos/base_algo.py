"""
Base Class for the algorithms
"""


class BaseAlgo:
    def __init__(self, num_actions: int):
        """
        Initialises the algorithm
        """
        self.name: str = "BaseAlgo"
        self.num_actions: int = num_actions

    def update(self, reward):
        """
        Updates the algorithm's internal state
        """
        raise NotImplementedError

    def current_policy(self):
        """
        Returns a policy
        """
        raise NotImplementedError

    def current_values(self):
        """
        Returns the action values
        """
        raise NotImplementedError

    def reset(self):
        """
        resets algorithms internal state
        """

        raise NotImplementedError
