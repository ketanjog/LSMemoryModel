from LSMemoryModel.envs.discrete_env import DiscreteEnv
from LSMemoryModel.algos.exp_weights import ExpWeights
from LSMemoryModel.constants.discrete import num_actions
import math

# Constants for the Environment
T = 10000

# Constants for the Algorithm
# To get optimum regret. Set learning rate to be sqrt(2 * log(num_actions) / T)
# learning_rate = math.sqrt(2 * math.log(num_actions) / T)
learning_rate = 1

algo = ExpWeights(learning_rate)
env = DiscreteEnv(T, algo)

env.train()



