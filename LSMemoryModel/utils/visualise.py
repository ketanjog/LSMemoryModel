"""
Class that takes probability distributions and lets you visualise them
in realtime
"""
from lib2to3.pytree import Base
import matplotlib.pyplot as plt
import numpy as np
from algos.base_algo import BaseAlgo


class Visualiser:
    def __init__(self, algo: BaseAlgo) -> None:

        self.algo: BaseAlgo = algo
        # TODO: Port to utils
        self.x = np.linspace(0, self.algo.num_actions, self.algo.num_actions)
        self.y = self.algo.current_policy()
        # to run GUI event loop
        plt.ion()

        # here we are creating sub plots
        self.figure, self.ax = plt.subplots(figsize=(10, 8))
        (self.visual,) = self.ax.plot(self.x, self.y)

        # setting title
        plt.title("Probabilities Over Actions", fontsize=20)

        # setting x-axis label and y-axis label
        plt.xlabel("Actions")
        plt.ylabel("Probability")
        plt.show(block=False)

    def render(self):
        # updating data values
        self.visual.set_xdata(self.x)
        new_y = self.algo.current_policy()
        self.visual.set_ydata(new_y)
        self.ax.relim()
        self.ax.autoscale_view(True, True, True)

        # drawing updated values
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

        # Pause is needed to render the image
        plt.pause(0.05)
