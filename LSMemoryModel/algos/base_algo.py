"""
Base Class for the algorithms
"""

class BaseAlgo:
    def __init__(self, run_id=None):
        """
        Initialises the algorithm
        """
        self.run_id = run_id
        self.name = "BaseAlgo"

    def choose_action(self, actions):
        """
        Returns the action to be taken
        """
        raise NotImplementedError

    def update(self):
        """
        Updates the algorithm's internal state
        """
        raise NotImplementedError