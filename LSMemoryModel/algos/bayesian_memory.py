"""
An Long Term Memory that stores contexts as distributions over action values
"""
from LSMemoryModel.algos.base_algo import BaseAlgo
from LSMemoryModel.utils.math import sample_normal
from collections import defaultdict
from typing import DefaultDict
import numpy as np
import torch


class BayesMemory(BaseAlgo):
    def __init__(self, num_actions: int):
        super().__init__(num_actions)

        # The set of stored contexts (world model)
        self.context_values: DefaultDict[int, dict()] = defaultdict()

        # The context the Agent thinks we are in
        self.present_context = None

        # The action the agent took
        self.action = None

    def update(self, reward, action, context):
        # print(self.context_values)

        self.context_values[context]["mean"][action] = self.context_values[context][
            "mean"
        ][action] + reward * self.context_values[context]["variance"][action] / (
            1 + self.context_values[context]["variance"][action]
        )

        self.context_values[context]["variance"][action] = self.context_values[context][
            "variance"
        ][action] / (1 + self.context_values[context]["variance"][action])

    def current_values(self, context):
        return self.context_values[context]

    def thompson_sample_each_context(self):
        action_by_context = torch.zeros((len(self.context_values), 2))

        for i in range(len(self.context_values)):
            mean = self.context_values[i]["mean"]
            variance = self.context_values[i]["variance"]
            # print(mean[0])
            # print("LOOK HERE")

            draws = torch.Tensor(
                [
                    sample_normal(
                        mean[j],
                        variance[j],
                    )
                    for j in range(self.num_actions)
                ]
            )

            best_action = torch.argmax(draws).item()

            action_by_context[i - 1][0] = best_action
            action_by_context[i - 1][1] = draws[best_action]

        return action_by_context
