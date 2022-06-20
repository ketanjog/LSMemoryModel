"""
Base Class for the algorithms
"""


class BaseAlgo:
    def __init__(self):
        """
        Initialises the algorithm
        """
        self.name = "BaseAlgo"

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

    def reset(self):
        """
        resets algorithms internal state
        """

        raise NotImplementedError
