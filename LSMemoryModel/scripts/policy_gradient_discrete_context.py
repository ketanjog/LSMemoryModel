from LSMemoryModel.envs.discrete_env import DiscreteEnv
from LSMemoryModel.agents.ltm_agent import LTMAgent

# Constants for the Environment
T = 5000

num_contexts = 5
num_actions = 100

context_epsilon = 0.01
action_positive_epsilon = 0.01
action_negative_epsilon = 0


stm_learning_rate = 0.5
learning_rate_ratio = 0.1
ltm_learning_rate = learning_rate_ratio * stm_learning_rate

stm_beta = 1
ltm_beta = 5

agent = LTMAgent(
    stm_learning_rate=stm_learning_rate,
    ltm_learning_rate=ltm_learning_rate,
    stm_beta=stm_beta,
    ltm_beta=ltm_beta,
    num_actions=num_actions,
)


env = DiscreteEnv(
    T=T,
    agent=agent,
    context_epsilon=context_epsilon,
    action_postive_epsilon=action_negative_epsilon,
    action_negative_epsilon=action_negative_epsilon,
    num_actions=num_actions,
    num_contexts=num_contexts,
    visual=False,
)
env.train()
