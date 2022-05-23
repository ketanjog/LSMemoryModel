from LSMemoryModel.algos.base_algo import BaseAlgo
from LSMemoryModel.constants.discrete import num_actions
import torch
from torch.distributions.categorical import Categorical
import numpy as np
import math


class PolicyGradient(BaseAlgo):
    def __init__(self, stm_learning_rate, run_id=None):
        super().__init__(run_id)
        self.name = "policy_gradient"

        # Short Term Memory Weights (Try random seeding)
        self.stm_weights = torch.ones(num_actions)
        # self.stm_weights /= torch.sum(self.stm_weights)

        # Probabilities of taking each action
        self.action_probs = torch.ones(num_actions)
        self.action_probs = torch.nn.functional.normalize(self.action_probs,dim=0,p=1)


        # STM Learning Rate        
        self.stm_learning_rate = stm_learning_rate


        self.last_action = None


    
    def choose_action(self, context):
        """
        Chooses the action with the maximum weight.
        If there are multiple max actions, a max action is
        randomly sampled
        """
        
        self.last_action = Categorical(self.action_probs).sample()

        return self.last_action

    def update(self, reward):
        """
        Updates the weight of the chosen action
        using the exponential learning rule.
        """

        def _I(a: int) -> int:
            indicator = torch.zeros(num_actions)
            indicator[a] = 1
            return indicator

        # Update the value function
        self.stm_weights += self.stm_learning_rate * reward * (_I(self.last_action) - self.action_probs)

        # Update the action probabilities
        self.action_probs = torch.nn.functional.normalize(torch.exp(self.stm_weights), dim=0,p=1)

class DualPolicyGradient(PolicyGradient):
    def __init__(self, stm_learning_rate, ltm_learning_rate, run_id=None):
        super().__init__(stm_learning_rate, run_id)
        self.name = "policy_gradient"

        # LTM Weights (Try random seeding)
        self.ltm_weights = torch.ones(num_actions)
        # self.stm_weights /= torch.sum(self.stm_weights)

        # Probabilities of taking each action
        self.s_action_probs = torch.ones(num_actions)
        self.s_action_probs = torch.nn.functional.normalize(self.s_action_probs,dim=0,p=1)

        self.l_action_probs = torch.ones(num_actions)
        self.l_action_probs = torch.nn.functional.normalize(self.l_action_probs,dim=0,p=1)

        self.action_probs = torch.ones(num_actions)
        self.action_probs = torch.nn.functional.normalize(self.action_probs,dim=0,p=1)


        # STM Learning Rate        
        self.ltm_learning_rate = stm_learning_rate


        self.last_action = None


    
    def choose_action(self, context):
        """
        Chooses the action with the maximum weight.
        If there are multiple max actions, a max action is
        randomly sampled
        """
        self.action_probs = torch.nn.functional.normalize(self.s_action_probs*self.l_action_probs,dim=0,p=1)

        self.last_action = Categorical(self.action_probs).sample()

        return self.last_action

    def update(self, reward):
        """
        Updates the weight of the chosen action
        using the exponential learning rule.
        """

        def _I(a: int) -> int:
            indicator = torch.zeros(num_actions)
            indicator[a] = 1
            return indicator

        # Update the value function
        self.stm_weights += self.stm_learning_rate * reward * (_I(self.last_action) - self.action_probs)
        self.ltm_weights += self.ltm_learning_rate * reward * (_I(self.last_action) - self.action_probs)

        # Update the action probabilities
        self.s_action_probs = torch.nn.functional.normalize(torch.exp(self.stm_weights), dim=0,p=1)
        self.l_action_probs = torch.nn.functional.normalize(torch.exp(self.ltm_weights), dim=0,p=1)

