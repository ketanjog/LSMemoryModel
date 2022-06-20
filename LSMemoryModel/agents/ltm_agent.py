"""
Agent that uses LTM  (slower learning process)
"""

from LSMemoryModel.agents.base_agent import BaseAgent
from LSMemoryModel.algos.policy_gradient import PGAlgo
from typing import DefaultDict
from collections import defaultdict
import torch
from LSMemoryModel.utils.math import sample


class LTMAgent(BaseAgent):
    def __init__(
        self,
        stm_learning_rate,
        ltm_learning_rate,
        stm_beta,
        ltm_beta,
        num_actions,
        run_id=None,
    ):
        super().__init__(run_id)

        # Define 2 systems LTM and STM
        self.systems: DefaultDict[str, PGAlgo] = defaultdict(PGAlgo)

        self.systems["STM"] = PGAlgo(
            learning_rate=stm_learning_rate, num_actions=num_actions, beta=stm_beta
        )
        self.systems["LTM"] = PGAlgo(
            learning_rate=ltm_learning_rate, num_actions=num_actions, beta=ltm_beta
        )

        # Save the action taken in this time step
        self.action = None

    def choose_action(self, context):
        """
        We are not allowed to see the context so disregard it
        Combine the two distributions and sample from them
        """
        # Multiply and normalise
        joint_distribution = torch.nn.functional.normalize(
            self.systems["STM"].current_policy() * self.systems["LTM"].current_policy(),
            dim=0,
            p=1,
        )

        # Put this in try loop in case of overflow
        try:
            self.action = sample(joint_distribution)
        except:
            pass

        return self.action
