"""
A class that produces context of a number c between [0, 2, ..., C-1]
"""

import numpy as np

from LSMemoryModel.constants.discrete import context_epsilon, action_epsilon
from LSMemoryModel.data.discrete_context import DISCRETE_CONTEXT
from LSMemoryModel.envs.base_env import BaseEnv


class DiscreteEnv(BaseEnv):
    def __init__(self, T, algo):
        super().__init__(T)
        self.data = DISCRETE_CONTEXT
        self.algo = algo
        self.reset_context()
        self.rewards = []
        self.cum_rewards = [0]
        self.t = 0

    def reset_context(self):
        """
        Randomly choose the context
        """
        _ = {"context": 0, "optimal_action": 1}

        idx = np.random.randint(len(DISCRETE_CONTEXT[_["context"]]))
        self.context = self.data[_["context"]][idx]
        self.optimal_action = self.data[_["optimal_action"]][idx]

    def step(self, action):
        """
        Returns a reward of 1 if action is optimal action
        If action is not optimal, returns reward of 1 with probability action_epsilon
        Returns a reward of 0 otherwise
        """
        self.t += 1

        if max(np.random.uniform(0,1), 1 if self.optimal_action == action else 0) > (1-action_epsilon):
            return 1
        
        return 0

    def update(self, r):
        """
        At every timestep, context changes with probability context_epsilon
        """
        if np.random.uniform(0, 1) < context_epsilon:
            self.reset_context()

        self.rewards.append(r)
        self.cum_rewards.append(r + self.cum_rewards[-1])

    def reset(self):
        self.t = 0
