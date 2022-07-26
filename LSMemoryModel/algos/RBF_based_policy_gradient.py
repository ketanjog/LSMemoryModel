"""
But how do I do the comparisions if there is no distribution on either side? 

The old way? Yee RBF Kernel over L1
"""
from LSMemoryModel.algos.base_algo import BaseAlgo
from LSMemoryModel.utils.math import onehot
from collections import defaultdict
from typing import DefaultDict
import torch


class RBFPGAlgo(BaseAlgo):
    def __init__(self, num_actions: int, learning_rate: float):
        super().__init__(num_actions)

        self.learning_rate = learning_rate

        # The set of stored contexts (world model)
        self.context_values: torch.Tensor = torch.zeros(num_actions)

        # The context the Agent thinks we are in
        self.present_context = None

        # The action the agent took
        self.action = None

    def update(self, reward, action, context):
        self.action = action
        self.present_context = context

        # Update the value function
        self.context_values[context] += (
            self.learning_rate
            * (reward - self.context_values[context])
            * onehot(self.action, self.num_actions)
        )

    def current_values(self, context):
        return self.context_values[context]

    def add_context(self, new_context):
        self.context_values = torch.vstack(self.context_values, new_context)
