from LSMemoryModel.algos.base_algo import BaseAlgo
from LSMemoryModel.constants.discrete import num_actions
import torch
import numpy as np
import math


class ExpWeights(BaseAlgo):
    def __init__(self, learning_rate, run_id=None):
        super().__init__(run_id)
        self.name = "ExpWeights"

        # Normalised weights vector
        self.weights = torch.ones(num_actions)
        self.weights = self.weights / torch.sum(self.weights)

        
        self.learning_rate = learning_rate
        self.last_action = None


    
    def choose_action(self, context):
        """
        Chooses the action with the maximum weight.
        If there are multiple max actions, a max action is
        randomly sampled
        """

        max_inds, = torch.where(self.weights == self.weights.max())
        
        self.last_action = np.random.choice(max_inds)

        return self.last_action

    def update(self, reward):
        """
        Updates the weight of the chosen action
        using the exponential learning rule.
        """
        self.weights[self.last_action] *= math.exp(-self.learning_rate*(1- reward))
        #self.weights[self.last_action] *= math.exp(self.learning_rate*(reward))
        
        # Renormalise
        self.weights /= torch.sum(self.weights)
        

        
                