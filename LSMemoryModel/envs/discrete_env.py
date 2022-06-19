"""
A class that produces context of a number c between [0, 2, ..., C-1]
"""

import numpy as np
from LSMemoryModel.data.discrete_context import get_discrete_context
from LSMemoryModel.envs.base_env import BaseEnv
from LSMemoryModel.algos.policy_gradient import PolicyGradient



class DiscreteEnv(BaseEnv):
    def __init__(
        self, 
        T, 
        algo, 
        context_epsilon, 
        action_postive_epsilon, 
        action_negative_epsilon, 
        num_actions, 
        num_contexts, 
        visual):
        super().__init__(T, num_actions, visual)
        self.data = get_discrete_context(num_actions, num_contexts)
        self.algo = algo
        self.reset_context()
        self.rewards = []
        self.cum_rewards = [0]
        self.t = 0
        self.action_postive_epsilon = action_postive_epsilon
        self.action_negative_epsilon = action_negative_epsilon
        self.context_epsilon = context_epsilon

        if isinstance(self.algo, PolicyGradient):
            self.reward_type = {"OPT" : 1, "OTHER" : -1}
        else:
            self.reward_type = {"OPT" : 1, "OTHER" : 0}

        self.prob_dist = self.algo.action_probs

    def reset_context(self):
        """
        Randomly choose the context
        """
        _ = {"context": 0, "optimal_action": 1}

        idx = np.random.randint(len(self.data[_["context"]]))
        self.context = self.data[_["context"]][idx]
        self.optimal_action = self.data[_["optimal_action"]][idx]
        #print("CHANGE")

    def step(self, action):
        """
        Returns a reward of 1 if action is optimal action
        If action is not optimal, returns reward of 1 with probability action_epsilon
        Returns a reward of 0 otherwise
        """
        self.t += 1

        # if self.t % 5 == 0:
        #         print("prob: " + str(self.algo.print))

        if self.optimal_action == action:
            if np.random.uniform(0,1) > self.action_negative_epsilon:
                return self.reward_type["OPT"]
            else:
                return self.reward_type["OTHER"]

        else:
            if np.random.uniform(0,1) > self.action_postive_epsilon:
                return self.reward_type["OTHER"]
            else:
                return self.reward_type["OPT"]
                



        # if max(np.random.uniform(0,1), 1 if self.optimal_action == action else 0) >= (1-self.action_epsilon):
        #     return self.reward_type["OPT"]
        
        # return self.reward_type["OTHER"]

    def update(self, r):
        """
        At every timestep, context changes with probability context_epsilon
        """
        if np.random.uniform(0, 1) < self.context_epsilon:
            self.reset_context()

        self.rewards.append(r)
        self.cum_rewards.append(r + self.cum_rewards[-1])

    def reset(self):
        self.t = 0
