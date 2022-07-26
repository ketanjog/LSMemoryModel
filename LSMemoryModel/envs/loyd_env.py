"""
Creates an environment as described in the Loyd Leslie model

https://doi.org/10.1098/rsif.2013.0069

"""
import numpy as np
from LSMemoryModel.data.loyd_context import get_loyd_context
from LSMemoryModel.envs.base_env import BaseEnv
from LSMemoryModel.algos.policy_gradient import PGAlgo
from LSMemoryModel.agents.base_agent import BaseAgent
from LSMemoryModel.utils.math import sample
import torch


class Context:
    def __init__(self, name) -> None:
        self.count: int = 0
        self.name: str = name


class EnvContext(Context):
    def __init__(self, action_reward_distribution, name) -> None:
        self.count: int = 0
        self.action_reward_distribution: torch.Tensor = action_reward_distribution
        self.name: str = name


class LoydEnv(BaseEnv):
    def __init__(
        self,
        T: int,
        agent: BaseAgent,
        action_reward_distribution: torch.Tensor,
        num_actions: int,
        visual: bool,
        context_epsilon: float = 0.075,  # Default value in the paper
        crp_alpha: int = 1,  # Default value in the paper
    ):
        """
        T: (int) number of rounds to run the simulation

        agent: (BaseAgent) the agent interacting with the environment

        context_epsilon: (float) the probability of switching contexts

        context_action_reward_distribution: (torch.Tensor) a matrix
                            of shape |A| x 2, where each action
                            has a tuple describing the mean reward and the
                            probability of recieving it. New contexts
                            will scramble this structure over actions.

        num_actions: (int) number of actions available to the agent

        visual: (bool) possibly deprecated. Shows a visual action value
                            distribution of agent during training
        """

        super().__init__(T, num_actions, visual)

        # Parametrizes the environment
        self.data = get_loyd_context(num_actions, action_reward_distribution)
        self.context_epsilon = context_epsilon
        self.crp_alpha = crp_alpha

        # Agent
        self.agent = agent

        # Saving values to look at later
        self.rewards = []
        self.cum_rewards = [0]
        self.t = 0
        self.contexts = None
        self.current_context_idx = None

        self.reward_type = {"OPT": 1, "OTHER": 0}

        # Initialise contexts
        self.initialise_environment()

    def initialise_environment(self):
        self.reset_context()

    def reset_context(self):
        """
        Randomly choose the context
        """
        # Confirm shape
        assert len(self.data) == self.num_actions

        # timestep 0, return the first default context
        if self.contexts == None:

            self.contexts.append(
                EnvContext(
                    action_reward_distribution=self.data, name=len(self.contexts) + 1
                )
            )

            # Set current context to
            self.current_context_idx = len(self.contexts)

        # Chinese Restaurant Process
        else:

            # Set weights for CRP sampling
            crp_weights = [i.count for i in self.contexts]
            crp_weights.append(self.crp_alpha)
            crp_weights = crp_weights / (self.t - 1 + self.crp_alpha)
            crp_weights = torch.nn.functional.normalize(
                crp_weights,
                dim=0,
                p=1,
            )

            # Choose using Chinese Restaurant Process
            self.current_context_idx = sample(crp_weights)

            # If we sample a new context, add it to the contexts list
            if self.current_context_idx == len(self.contexts) + 1:
                self.contexts.append(
                    EnvContext(
                        action_reward_distribution=self.data[
                            torch.randperm(len(self.data))
                        ],
                        name=self.current_context_idx,
                    )
                )

    def step(self, action):
        """
        Returns a reward based on the reward structure of the current context
        """
        # Increment time
        self.t += 1

        # better nomenclature based on action_reward tensor
        REWARD = 0
        PROBABILITY = 1

        reward = self.contexts[self.current_context_idx][action][
            REWARD
        ] * torch.distributions.bernoulli.Bernoulli(
            torch.Tensor(self.contexts[self.current_context_idx][action][PROBABILITY])
        )

    def update(self, r):
        """
        At every timestep, context changes with probability context_epsilon
        """
        if np.random.uniform(0, 1) < self.context_epsilon:
            self.reset_context()

        # Increment count on the context
        self.contexts(self.current_context_idx).count += 1

        self.rewards.append(r)
        self.cum_rewards.append(r + self.cum_rewards[-1])

    def reset(self):
        self.t = 0
