from LSMemoryModel.algos.base_algo import BaseAlgo
from LSMemoryModel.constants.discrete import num_actions
import torch
import numpy as np
import math


class DualWeights(BaseAlgo):
    def __init__(self, stm_learning_rate, ltm_learning_rate, run_id=None):
        super().__init__(run_id)
        self.name = "DualWeights"

        # Normalised weights vector
        self.ltm_weights = torch.ones(num_actions)
        self.ltm_weights /= torch.sum(self.ltm_weights)

        self.stm_weights = torch.ones(num_actions)
        self.stm_weights /= torch.sum(self.stm_weights)

        
        self.ltm_learning_rate = ltm_learning_rate
        self.stm_learning_rate = stm_learning_rate
        self.last_action = None
        self.prev_reward = 1


    
    def choose_action(self, context):
        """
        Chooses the action with the maximum weight.
        If there are multiple max actions, a max action is
        randomly sampled
        """

        # Multiplicative agreement
        weights = self.ltm_weights * self.stm_weights

        # Additive agreement
        # weights = self.ltm_weights + self.stm_weights

        # Max agreement
        # weights = torch.maximum(self.stm_weights, self.ltm_weights)

        # Max gated agreement
        # weights = torch.maximum(self.stm_weights, (1- self.prev_reward) * self.ltm_weights)


        max_inds, = torch.where(weights == weights.max())
        
        self.last_action = np.random.choice(max_inds)
        
        return self.last_action

    def update(self, reward):
        """
        Updates the weight of the chosen action
        using the exponential learning rule.
        """
        self.ltm_weights[self.last_action] *= math.exp(-self.ltm_learning_rate*(1-reward))
        self.stm_weights[self.last_action] *= math.exp(-self.stm_learning_rate*(1-reward))
            
        
        # Renormalise 
        self.ltm_weights /= torch.sum(self.ltm_weights)
        self.stm_weights /= torch.sum(self.stm_weights)
        self.prev_reward = reward
        

        
                