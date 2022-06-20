"""
Base Class for the Environments
"""
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import time
from agents.base_agent import BaseAgent


class BaseEnv:
    def __init__(
        self, T: int, num_actions: int, agent: BaseAgent = None, visual: bool = False
    ):
        """
        Initializes the environment
        """
        self.visual: bool = visual
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
            # TODO: Port to utils
            x = np.linspace(0, self.num_actions, self.num_actions)
            y = self.algo.action_probs
            # to run GUI event loop
            plt.ion()

            # here we are creating sub plots
            figure, ax = plt.subplots(figsize=(10, 8))
            (line1,) = ax.plot(x, y)

            # setting title
            plt.title("Probabilities Over Actions", fontsize=20)

            # setting x-axis label and y-axis label
            plt.xlabel("Actions")
            plt.ylabel("Probability")
            plt.show(block=False)

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

                # updating data values
                line1.set_xdata(x)
                new_y = self.algo.action_probs
                line1.set_ydata(new_y)
                ax.relim()
                ax.autoscale_view(True, True, True)

                # drawing updated values
                figure.canvas.draw()
                figure.canvas.flush_events()

                # Pause is needed to render the image
                plt.pause(0.05)
