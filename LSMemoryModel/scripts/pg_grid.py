"""
Gridsearch script for Policy Gradient method.
"""

from LSMemoryModel.envs.discrete_env import DiscreteEnv
from LSMemoryModel.algos.policy_gradient import PolicyGradient, DualPolicyGradient
from LSMemoryModel.constants.discrete import num_actions
import math

# Constants for the Environment
T = 10000
num_runs = 20
context_epsilon_params = [0, 0.01, 0.05, 0.1, 0.2] 
actions_epsilon = 0.01
num_actions_params = [10, 100, 1000, 10000]
num_contexts_params = [1, 2, 5, 10]

# Constants for the Algorithm
stm_learning_rate_params = [0.001, 0.01, 0.1, 1, 10]


algo = PolicyGradient()
#algo = DualPolicyGradient(stm_learning_rate, ltm_learning_rate)
env = DiscreteEnv()

env.train()




