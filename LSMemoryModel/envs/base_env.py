"""
Base Class for the Environments
"""
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import time
from LSMemoryModel.agents.base_agent import BaseAgent
from LSMemoryModel.utils.visualise import Visualiser


class BaseEnv:
    def __init__(
        self,
        T: int,
        num_actions: int,
        agent: BaseAgent = None,
        visual: bool = False,
        system: str = None,
    ):
        """
        Initializes the environment
        """
        self.visual: bool = visual
        self.to_visualise: str = system
        self.name: str = "BaseEnv"
        self.T: int = T
        self.num_actions: int = num_actions
        self.agent: BaseAgent = agent
        self.context = None

    def update(self) -> None:
        """
        Updates the environments internal state
        """
        raise NotImplementedError

    def step(self, action) -> float:
        """
        Returns the reward for the action taken
        """
        raise NotImplementedError

    def reset(self) -> None:
        """
        Resets the environment
        """
        raise NotImplementedError

    def train(self) -> None:
        """
        Trains the algorithm
        """
        if self.visual:
            visualiser = Visualiser(algo=self.agent.systems[self.to_visualise])

        pbar = tqdm(total=self.T)

        # Rum the environment for T rounds
        for _ in range(self.T):

            # Provide context to the agent. Collect its action
            action = self.agent.choose_action(self.context)

            # Perform the action in the environment, get reward
            r = self.step(action)

            # Report the reward to the agent
            self.agent.update(r)

            # Based on the reward/action, update the environment state.
            self.update(r)

            # print the reward
            pbar.set_description(f"Reward/time: {self.cum_rewards[-1]/self.t:.2f}")
            pbar.update(1)

            # If visual, show the action probabilities of the agent
            if self.visual:
                visualiser.render()
