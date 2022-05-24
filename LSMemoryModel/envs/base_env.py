"""
Base Class for the Environments
"""
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import time 



class BaseEnv:
    def __init__(self, T, num_actions, visual=False):
        """
        Initializes the environment
        """
        self.visual = visual
        self.name = "BaseEnv"
        self.T = T
        self.num_actions = num_actions

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
        if self.visual:
            # TODO: Port to utils
            x = np.linspace(0, self.num_actions, self.num_actions)
            y = self.algo.action_probs
            # to run GUI event loop
            plt.ion()
            
            # here we are creating sub plots
            figure, ax = plt.subplots(figsize=(10, 8))
            line1, = ax.plot(x, y)
            
            # setting title
            plt.title("Probabilities Over Actions", fontsize=20)
            
            # setting x-axis label and y-axis label
            plt.xlabel("Actions")
            plt.ylabel("Probability")
            plt.show(block=False)

        pbar = tqdm(total=self.T)
        for _ in range(self.T):
            action = self.algo.choose_action(self.context)
            r = self.step(action)
            self.algo.update(r)
            self.update(r)

            # print the reward
            pbar.set_description(f"Reward/time: {self.cum_rewards[-1]/self.t:.2f}")
            pbar.update(1)

            
            if self.visual:
                # updating data values
                line1.set_xdata(x)
                new_y = self.algo.action_probs
                line1.set_ydata(new_y)
                ax.relim() 
                ax.autoscale_view(True,True,True) 
                # drawing updated values
                figure.canvas.draw()
                figure.canvas.flush_events()
                #time.sleep(0.1)
                plt.pause(0.05)
