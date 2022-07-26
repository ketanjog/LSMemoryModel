"""
Agent that utilises a LTM of a set of previously seen
contexts, formally also a Bayes LTM with
clamped variance
"""

from LSMemoryModel.agents.base_agent import BaseAgent
from LSMemoryModel.algos.base_algo import BaseAlgo
from LSMemoryModel.algos.RBF_based_policy_gradient import RBFPGAlgo
from LSMemoryModel.algos.policy_gradient import PGAlgo
from LSMemoryModel.utils.math import rbf_l1, sample
import torch
from typing import DefaultDict
from collections import defaultdict


class BayesAgent(BaseAgent):
    def __init__(
        self,
        num_actions,
        stm_learning_rate,
        ltm_learning_rate,
        radius,
        threshold,
        run_id=None,
    ):
        super().__init__(run_id)

        # Define 2 systems LTM and STM
        self.systems: DefaultDict[str, BaseAlgo] = defaultdict(BaseAlgo)

        self.systems["STM"] = PGAlgo(
            learning_rate=stm_learning_rate, num_actions=num_actions, beta=1
        )
        self.systems["LTM"] = RBFPGAlgo(
            num_actions=num_actions, learning_rate=ltm_learning_rate
        )

        self.num_actions = num_actions
        self.similarity_index = None

        self.action = None
        self.context = None

        self.NEW_CONTEXT_THRESHOLD = threshold
        self.radius = radius

    def update(self, reward):
        """
        Updates the internal states
        of its systems
        """
        self.systems["STM"].update(reward, self.action)
        self.systems["LTM"].update(reward, self.action, self.context)

    def choose_action(self):
        num_contexts = len(self.systems["LTM"].context_values)

        if num_contexts == 0:
            self.systems["LTM"].add_context(self.systems["STM"].current_values())
            num_contexts = 1

        self.similarity_index = torch.Tensor(
            [
                rbf_l1(
                    self.systems["STM"].current_values(),
                    self.systems["LTM"].current_values(i),
                    radius=self.radius,
                ).item()
                for i in range(num_contexts)
            ]
        )

        if min(self.similarity_index) > self.NEW_CONTEXT_THRESHOLD:
            # add stm to ltm
            self.systems["LTM"].add_context(self.systems["STM"].current_values())
            torch.cat((self.similarity_index, torch.Tensor([0])), dim=0)

        # Sample a context based on calculated similarity:
        # context = sample(
        #     torch.nn.functional.normalize(self.similarity_index, p=1, dim=0)
        # )

        best_action_by_context = self.systems["LTM"].thompson_sample_each_context()

        self.context = torch.argmax(
            best_action_by_context.T[1] / self.dissimilarity_index
        ).item()
        self.action = int(best_action_by_context[self.context][0].item())

        return self.action
