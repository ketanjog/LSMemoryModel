from LSMemoryModel.algos.base_algo import BaseAlgo
import torch
from LSMemoryModel.utils.math import onehot, sample, softmax


class PGAlgo(BaseAlgo):
    def __init__(self, learning_rate, num_actions, beta):

        super().__init__(num_actions)
        self.name = "policy_gradient"

        # Short Term Memory Weights (Try random seeding)
        self.weights = torch.zeros(self.num_actions)

        # STM Learning Rate
        self.learning_rate = learning_rate

        self.last_action = None
        self.beta = beta

    def update(self, reward, last_action):

        """
        Updates the weight of the chosen action
        using the exponential learning rule.
        """
        self.last_action = last_action

        # Update the value function
        self.weights += (
            self.learning_rate
            * (reward - self.weights)
            * onehot(self.last_action, self.num_actions)
        )

    def current_policy(self):
        return softmax(self.weights, self.beta)
