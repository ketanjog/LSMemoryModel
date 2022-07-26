"""
Implementation of a bayesian learning model, 

as described in https://doi.org/10.1098/rsif.2013.0069

"""

from LSMemoryModel.agents.base_agent import BaseAgent
from LSMemoryModel.algos.policy_gradient import PGAlgo
from typing import DefaultDict
from collections import defaultdict
import torch
from LSMemoryModel.utils.math import sample


class LoydAgent(BaseAgent):
    def __init__(
        self,
        num_actions,
        m_0=0,
        k_0=1,
        A_0=1,
        B_0=1,
        context_epsilon=0.075,
        crp_alpha=1,
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

    def choose_action(self):
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
