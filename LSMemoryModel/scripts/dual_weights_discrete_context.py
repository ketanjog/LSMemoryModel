from LSMemoryModel.envs.discrete_env import DiscreteEnv
from LSMemoryModel.algos.dual_weights import DualWeights
from LSMemoryModel.constants.discrete import num_actions
import math

# Constants for the Environment
T = 10000

# Constants for the Algorithm
# To get optimum regret. Set learning rate to be sqrt(2 * log(num_actions) / T)
stm_learning_rate = math.sqrt(2 * math.log(num_actions) / T)
ltm_learning_rate = stm_learning_rate / 10

algo = DualWeights(stm_learning_rate, ltm_learning_rate)
env = DiscreteEnv(T, algo)

env.train()



