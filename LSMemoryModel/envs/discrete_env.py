'''
A class that produces context of a number c between [0, 2, ..., C-1]
'''

from LSMemoryModel.envs.base_env import BaseEnv
from LSMemoryModel.data.discrete_context import DISCRETE_CONTEXT
from LSMemoryModel.constants.discrete_context import epsilon
import numpy as np

class DiscreteEnv(BaseEnv):
    def __init__(self, T):
        super().__init__(T)
        self.data = DISCRETE_CONTEXT
        self.reset_context()
        self.rewards = []
        self.cum_rewards = []

    def reset_context(self):
        """
        Randomly choose the context
        """
        _ = {"context" : 0, "optimal_action" : 1}

        idx = np.random.randint(len(DISCRETE_CONTEXT[_["context"]]))
        self.context = self.data[_["context"]][idx]
        self.optimal_action = self.data[_["optimal_action"]][idx]

    def step(self, action):
        self.t += 1
        return 1 if self.optimal_action == action else 0

    def update(self, r):
        if np.random.uniform(0,1) < epsilon:
            self.reset_context()

        self.rewards.append(r)
        self.cum_rewards.append(r + self.cum_rewards[-1])

    
    def reset(self):
        self.t = 0

        
        
    

