"""
The Bayesian Model as described in the Loyd Leslie paper
"""
import torch
from LSMemoryModel.algos.base_algo import BaseAlgo
from LSMemoryModel.envs.loyd_env import Context


class BayesContext(Context):
    def __init__(self, name, num_actions) -> None:
        super().__init__(name)

        # all params to create
        class NormalGamma:
            def __init__(self) -> None:
                # the number of observations assigned to the context–action pair
                self.n = 0
                # the mean of the observed rewards assigned to the context–action pair
                self.mean_reward = 0

                # context–action sum of squares
                self.sos = 0

            def update(self, reward):
                self.n += 1

                self.mean_reward = (
                    self.n / (self.n + 1)
                ) * self.mean_reward + reward / (self.n + 1)

                self.sos += (reward - self.mean_reward) ** 2

            def m_t(self, k_0, m_0):
                return (k_0 * m_0 + self.n * self.mean_reward) / (k_0 + self.n)

            def k_t(self, k_0):
                return k_0 + self.n

            def A_t(self, A_0):
                return A_0 + self.n / 2

            def B_t(self, B_0, k_0, m_0):
                return (
                    B_0
                    + 0.5 * self.sos
                    + (k_0 * self.n(self.mean_reward - m_0) ** 2) / (2(k_0 + self.n))
                )

        self.theta = [NormalGamma() for i in range(num_actions)]


class LoydAlgo(BaseAlgo):
    def __init__(
        self,
        num_actions,
        m_0=0,
        k_0=1,
        A_0=1,
        B_0=1,
        context_epsilon=0.075,
        crp_alpha=1,
    ):

        super().__init__(num_actions)
        self.name = "loyd"

        # Set params
        self.m_0 = m_0
        self.k_0 = k_0
        self.A_0 = A_0
        self.B_0 = B_0
        self.context_epsilon = context_epsilon
        self.crp_alpha = crp_alpha

        self.last_action = None
        self.current_context_idx = None
        self.contexts = None
        self.context_switch: bool = False

    def likelihood(self, reward, context, action):

        ca = self.contexts[context][action]

        shape = 2 * ca.A_t
        location = ca.m_t
        scale = (ca.B_t * (ca.k_t + 1)) / (ca.A_t * ca.k_t)

        likelihood = (
            torch.distributions.studentT.StudentT(df=shape, loc=location, scale=scale)
            .log_prob(reward)
            .exp()
        )

    def update(self, reward, last_action):

        """
        Updates the weight of the chosen action

        based on a Bayesian calculation
        """
        self.last_action = last_action

        if len(self.contexts == 0):
            self.contexts.append(
                BayesContext(name=len(self.contexts) + 1, num_actions=self.num_actions)
            )

            self.current_context_idx = len(self.contexts)

        # We have at least 1 context

    def current_policy(self):
        return softmax(self.weights, self.beta)

    def current_values(self):
        return self.weights
