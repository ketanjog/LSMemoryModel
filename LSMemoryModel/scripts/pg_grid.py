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
import sys

# Constants for the Environment
T = 5000
num_runs = 5

context_epsilon_params = [0, 0.01, 0.05, 0.1] 
action_epsilon_positive_params = [0.01, 0.05, 0.1]
action_epsilon_negative_params = [0.01, 0.05, 0.1]

num_actions_params = [10, 100, 1000]
num_contexts_params = [1, 2, 5, 10]

#stm_learning_rate_params = [0.001, 0.01, 0.1, 0.5]
stm_learning_rate_params = [0.1, 0.5]
# ltm_learning_rate_ratio_params = [0, 0.001, 0.01, 0.1]
ltm_learning_rate_ratio_params = [sys.argv[1]]

beta_stm_params = [0.1, 1, 10]
beta_ltm_ratio_params = [1, 2, 5]



# Constants for the Algorithm



def _name(vars: list()) -> str:
    string_vars = [str(var) for var in vars]
    return "_".join(string_vars)

eval_metrics = defaultdict(list)

for ltm_learning_rate_ratio in ltm_learning_rate_ratio_params:
    for stm_learning_rate in stm_learning_rate_params:
        print("Running stm = " + str(stm_learning_rate))
        for num_contexts in tqdm(num_contexts_params):
            for num_actions in num_actions_params:
                
                for context_epsilon in context_epsilon_params:
                    for action_epsilon_positive in action_epsilon_positive_params:
                        for action_epsilon_negative in action_epsilon_negative_params:
                            for beta_stm in beta_stm_params:
                                for beta_ltm_ratio in beta_ltm_ratio_params:

                                    ltm_learning_rate = stm_learning_rate * ltm_learning_rate_ratio
                                    beta_ltm = beta_stm * beta_ltm_ratio
                                

                                    name: str = _name([
                                                    stm_learning_rate,
                                                    ltm_learning_rate,
                                                    num_contexts, 
                                                    num_actions, 
                                                    context_epsilon,
                                                    action_epsilon_positive,
                                                    action_epsilon_negative,
                                                    beta_stm,
                                                    beta_ltm
                                                    ])

                                    for _run in range(num_runs):                       

                                        algo = DualPolicyGradient(
                                                            stm_learning_rate, 
                                                            ltm_learning_rate, 
                                                            num_actions, 
                                                            beta_stm, 
                                                            beta_ltm,
                                                            run_id=name 
                                                            )
                                        env = DiscreteEnv(
                                                            T, 
                                                            algo, 
                                                            context_epsilon, 
                                                            action_epsilon_positive, 
                                                            action_epsilon_negative, 
                                                            num_actions, 
                                                            num_contexts, 
                                                            visual=False
                                                            )
                                        env.train()
                                        
                                        eval_metrics[name].append(env.cum_rewards[-1])
                        

                                        env.reset()
                                    save_object(eval_metrics[name], name)





