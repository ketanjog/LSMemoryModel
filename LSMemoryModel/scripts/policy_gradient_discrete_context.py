from LSMemoryModel.envs.discrete_env import DiscreteEnv
from LSMemoryModel.algos.policy_gradient import PolicyGradient, DualPolicyGradient
from LSMemoryModel.constants.discrete import num_actions
import math

# Constants for the Environment
T = 10000
context_epsilon = 0.01
action_epsilon = 0.01
num_contexts = 5


# Constants for the Algorithm
# To get optimum regret. Set learning rate to be sqrt(2 * log(num_actions) / T)
# stm_learning_rate = math.sqrt(2 * math.log(num_actions) / T)
stm_learning_rate = 1
ltm_learning_rate = 0.01
num_actions = 100

algo = PolicyGradient(stm_learning_rate,num_actions)
#algo = DualPolicyGradient(stm_learning_rate, ltm_learning_rate)
env = DiscreteEnv(T, algo, context_epsilon, action_epsilon, num_actions, num_contexts)

env.train()




