from LSMemoryModel.envs.discrete_env import DiscreteEnv
from LSMemoryModel.algos.policy_gradient import PolicyGradient, DualPolicyGradient
from LSMemoryModel.constants.discrete import num_actions
import math

# Constants for the Environment
T = 1000
context_epsilon = 0.1
action_epsilon = 0
num_contexts = 5


# Constants for the Algorithm
# To get optimum regret. Set learning rate to be sqrt(2 * log(num_actions) / T)
# stm_learning_rate = math.sqrt(2 * math.log(num_actions) / T)
stm_learning_rate = 1
learning_rate_ratio = 0
ltm_learning_rate = learning_rate_ratio * stm_learning_rate
num_actions = 10
beta_stm = 1
beta_ltm = 1
beta = 1

#algo = PolicyGradient(stm_learning_rate,num_actions,beta)
algo = DualPolicyGradient(stm_learning_rate, ltm_learning_rate, num_actions, beta_stm, beta_ltm)
env = DiscreteEnv(T, algo, context_epsilon, action_epsilon, num_actions, num_contexts, visual=True)

env.train()




