"""
Gridsearch script for Policy Gradient method.
"""

from LSMemoryModel.envs.discrete_env import DiscreteEnv
from LSMemoryModel.agents.ltm_agent import LTMAgent
from LSMemoryModel.constants.discrete import num_actions
from LSMemoryModel.utils.save import save_object
from collections import defaultdict
from tqdm import tqdm
import sys

# Constants for the Environment
T = 100000
num_runs = 3

context_epsilon_params = [0.01]
action_epsilon_positive_params = [0.01]
action_epsilon_negative_params = [0.01]

# num_actions_params = [100, 500, 1000, 5000]
num_actions_params = [sys.argv[1]]
num_contexts_params = [10]

stm_learning_rate_params = [0.5]
ltm_learning_rate_ratio_params = [0, 0.01, 0.1]

beta_stm_params = [10, 50, 100]
beta_ltm_params = [10, 50, 100]


# Constants for the Algorithm


def _name(vars: list()) -> str:
    string_vars = [str(var) for var in vars]
    return "_".join(string_vars)


eval_metrics = defaultdict(list)
eval_time_series = defaultdict(list)

for ltm_learning_rate_ratio in ltm_learning_rate_ratio_params:
    for stm_learning_rate in stm_learning_rate_params:
        print("Running stm = " + str(stm_learning_rate))
        for num_contexts in tqdm(num_contexts_params):
            for num_actions in num_actions_params:

                for context_epsilon in context_epsilon_params:
                    for action_epsilon_positive in action_epsilon_positive_params:
                        for action_epsilon_negative in action_epsilon_negative_params:
                            for stm_beta in beta_stm_params:
                                for ltm_beta in beta_ltm_params:

                                    ltm_learning_rate = (
                                        stm_learning_rate * ltm_learning_rate_ratio
                                    )
                                    # beta_ltm = beta_stm * beta_ltm_ratio

                                    name: str = _name(
                                        [
                                            stm_learning_rate,
                                            ltm_learning_rate,
                                            num_contexts,
                                            num_actions,
                                            context_epsilon,
                                            action_epsilon_positive,
                                            action_epsilon_negative,
                                            stm_beta,
                                            ltm_beta,
                                        ]
                                    )

                                    time_series: str = _name(["time_series", name])

                                    for _run in range(num_runs):

                                        agent = LTMAgent(
                                            stm_learning_rate=stm_learning_rate,
                                            ltm_learning_rate=ltm_learning_rate,
                                            stm_beta=stm_beta,
                                            ltm_beta=ltm_beta,
                                            num_actions=num_actions,
                                        )
                                        env = DiscreteEnv(
                                            T,
                                            agent,
                                            context_epsilon,
                                            action_epsilon_positive,
                                            action_epsilon_negative,
                                            num_actions,
                                            num_contexts,
                                            visual=False,
                                        )
                                        env.train()

                                        eval_metrics[name].append(env.cum_rewards[-1])
                                        eval_time_series[time_series].append(
                                            env.cum_rewards
                                        )

                                        env.reset()
                                    save_object(eval_metrics[name], name)
                                    save_object(
                                        eval_time_series[time_series], time_series
                                    )
