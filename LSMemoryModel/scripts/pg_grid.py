"""
Gridsearch script for Policy Gradient method.
"""

from LSMemoryModel.envs.discrete_env import DiscreteEnv
from LSMemoryModel.algos.policy_gradient import PolicyGradient, DualPolicyGradient
from LSMemoryModel.constants.discrete import num_actions
from LSMemoryModel.utils.save import save_object
import math
from collections import defaultdict
from tqdm import tqdm

# Constants for the Environment
T = 10000
num_runs = 10
context_epsilon_params = [0, 0.01, 0.05, 0.1, 0.2] 
action_epsilon = 0.01
num_actions_params = [10, 100, 1000, 10000]
num_contexts_params = [1, 2, 5, 10]


# Constants for the Algorithm
stm_learning_rate_params = [0.001, 0.01, 0.1, 1]
beta_params = [0.01, 0.1, 1, 10]

def _name(vars: list()) -> str:
    string_vars = [str(var) for var in vars]
    return "_".join(string_vars)

eval_metrics = defaultdict(list)

for stm_learning_rate in stm_learning_rate_params:
    print("Running stm = " + str(stm_learning_rate))
    for num_contexts in tqdm(num_contexts_params):
        for num_actions in num_actions_params:
            for context_epsilon in context_epsilon_params:
                for beta in beta_params:
                    

                    name: str = _name([
                                    stm_learning_rate, 
                                    num_contexts, 
                                    num_actions, 
                                    context_epsilon,
                                    beta
                                    ])

                    for _run in range(num_runs):                       

                        algo = PolicyGradient(
                                            stm_learning_rate,
                                            num_actions,
                                            beta,
                                            run_id = name
                                            )
                        env = DiscreteEnv(
                                            T, 
                                            algo, 
                                            context_epsilon, 
                                            action_epsilon, 
                                            num_actions, 
                                            num_contexts,
                                            visual=False)
                        env.train()
                        
                        eval_metrics[name].append(env.cum_rewards[-1])
        

                        env.reset()
                    save_object(eval_metrics[name], name)





