"""
Base Class for the agents
"""

from algos.base_algo import BaseAlgo
from typing import DefaultDict
from collections import defaultdict


class BaseAgent:
    def __init__(self, run_id=None):
        """
        Initialises the algorithm
        """
        self.run_id: str = run_id
        self.name: str = "Base-Agent"

        self.systems: DefaultDict[str, BaseAlgo] = defaultdict(BaseAlgo)

    def choose_action(self, context):
        """
        Returns the action to be taken based on the outputs
        of its systems
        """

        raise NotImplementedError

    def update(self, reward):
        """
        Updates the internal states
        of its systems
        """

        for name, system in self.systems.items():
            system.update(reward)

    def reset(self):
        """
        Resets internal states
        of its systems
        """
        for system in self.systems:
            system.reset()
