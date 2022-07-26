"""
Agent that utilises a Bayesian LTM, formally also uses a Bayes LTM with
clamped variance
"""
from LSMemoryModel.agents.base_agent import BaseAgent
from LSMemoryModel.algos.base_algo import BaseAlgo
from LSMemoryModel.algos.bayesian_memory import BayesMemory
from LSMemoryModel.algos.policy_gradient import PGAlgo
from LSMemoryModel.utils.math import gaussianKL
from typing import DefaultDict
from collections import defaultdict
import torch


class BayesAgent(BaseAgent):
    def __init__(self, num_actions, stm_learning_rate, threshold, run_id=None):
        super().__init__(run_id)

        # Define 2 systems LTM and STM
        self.systems: DefaultDict[str, BaseAlgo] = defaultdict(BaseAlgo)

        self.systems["STM"] = PGAlgo(
            learning_rate=stm_learning_rate, num_actions=num_actions, beta=1
        )
        self.systems["LTM"] = BayesMemory(num_actions=num_actions)

        self.num_actions = num_actions
        self.dissimilarity_index = None

        self.action = None
        self.context = None
        self.NEW_CONTEXT_THRESHOLD = threshold

    def choose_action(self):
        num_contexts = len(self.systems["LTM"].context_values)

        # STM has a fixed variance per action value of 0.5

        stm_distribution = {
            "mean": self.systems["STM"].current_values(),
            "variance": torch.full((self.num_actions,), 0.5),
        }
        if num_contexts != 0:

            self.dissimilarity_index = torch.Tensor(
                [
                    gaussianKL(
                        stm_distribution,
                        self.systems["LTM"].current_values(i),
                    )
                    for i in range(num_contexts)
                ]
            )

            if min(self.dissimilarity_index) > self.NEW_CONTEXT_THRESHOLD:
                # add stm to ltm
                self.systems["LTM"].context_values[num_contexts] = stm_distribution
                self.action = torch.argmax(stm_distribution["mean"]).item()
                self.context = num_contexts
            else:

                best_action_by_context = self.systems[
                    "LTM"
                ].thompson_sample_each_context()

                self.context = torch.argmax(
                    best_action_by_context.T[1] / self.dissimilarity_index
                ).item()
                self.action = int(best_action_by_context[self.context][0].item())
        else:
            # add stm to ltm
            self.systems["LTM"].context_values[num_contexts] = stm_distribution
            self.action = torch.argmax(stm_distribution["mean"]).item()
            self.context = 0

        return self.action

    def update(self, reward):
        """
        Updates the internal states
        of its systems
        """
        self.systems["STM"].update(reward, self.action)
        self.systems["LTM"].update(reward, self.action, self.context)
