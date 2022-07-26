from torch import threshold
from LSMemoryModel.envs.discrete_env import DiscreteEnv
from LSMemoryModel.agents.bayes_agent import BayesAgent

# Constants for the Environment
T = 2000

num_contexts = 10
num_actions = 100

context_epsilon = 0.01
action_positive_epsilon = 0.01
action_negative_epsilon = 0


# Constants for Algo
stm_learning_rate = 0.5
threshold = 1


agent = BayesAgent(
    num_actions=num_actions, stm_learning_rate=stm_learning_rate, threshold=threshold
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


for _, i in agent.systems["LTM"].context_values.items():
    print(i["mean"])

print(len(agent.systems["LTM"].context_values))
