from LSMemoryModel.envs.discrete_env import DiscreteEnv
from LSMemoryModel.agents.ltm_agent import LTMAgent

# Constants for the Environment
T = 20000

num_contexts = 10
num_actions = 100

context_epsilon = 0.01
action_positive_epsilon = 0.01
action_negative_epsilon = 0.01


stm_learning_rate = 0.5
learning_rate_ratio = 0.0
ltm_learning_rate = learning_rate_ratio * stm_learning_rate

stm_beta = 10
ltm_beta = 10

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
