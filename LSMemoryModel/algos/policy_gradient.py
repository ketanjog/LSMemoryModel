from LSMemoryModel.algos.base_algo import BaseAlgo
import torch
from utils.math import onehot, sample, softmax


class PGAlgo(BaseAlgo):
    def __init__(self, learning_rate, num_actions, beta):

        super().__init__()
        self.name = "policy_gradient"
        self.num_actions = num_actions

        # Short Term Memory Weights (Try random seeding)
        self.weights = torch.ones(self.num_actions)

        # STM Learning Rate
        self.learning_rate = learning_rate

        self.last_action = None
        self.beta = beta

    def update(self, reward):
        """
        Updates the weight of the chosen action
        using the exponential learning rule.
        """

        # Update the value function
        self.weights += (
            self.learning_rate
            * (reward - self.weights)
            * onehot(self.last_action, self.num_actions)
        )

    def current_policy(self):
        return softmax(self.weights, self.beta)


class DualPolicyGradient(BaseAlgo):
    def __init__(
        self,
        stm_learning_rate,
        ltm_learning_rate,
        num_actions,
        beta_stm,
        beta_ltm,
        run_id=None,
    ):

        super().__init__(run_id)
        self.name = "dual_policy_gradient"

        self.num_actions = num_actions
        # Short Term Memory Weights (Try random seeding)
        self.stm_weights = torch.zeros(self.num_actions)

        # LTM Weights (Try random seeding)
        self.ltm_weights = torch.zeros(self.num_actions)

        # Probabilities of taking each action
        self.s_action_probs = torch.ones(self.num_actions)
        self.s_action_probs = torch.nn.functional.normalize(
            self.s_action_probs, dim=0, p=1
        )

        self.l_action_probs = torch.ones(self.num_actions)
        self.l_action_probs = torch.nn.functional.normalize(
            self.l_action_probs, dim=0, p=1
        )

        self.action_probs = torch.ones(self.num_actions)
        self.action_probs = torch.nn.functional.normalize(self.action_probs, dim=0, p=1)

        # STM Learning Rate
        self.ltm_learning_rate = ltm_learning_rate
        self.stm_learning_rate = stm_learning_rate

        self.last_action = None

        self.beta_ltm = beta_ltm
        self.beta_stm = beta_stm

    def choose_action(self, context):
        """
        Chooses the action with the maximum weight.
        If there are multiple max actions, a max action is
        randomly sampled
        """
        self.action_probs = torch.nn.functional.normalize(
            self.s_action_probs * self.l_action_probs, dim=0, p=1
        )

        self.last_action = Categorical(self.action_probs).sample()

        return self.last_action

    def update(self, reward):
        """
        Updates the weight of the chosen action
        using the exponential learning rule.
        """

        def _I(a: int) -> int:
            indicator = torch.zeros(self.num_actions)
            indicator[a] = 1
            return indicator

        # Update the value function
        # self.stm_weights += self.stm_learning_rate * reward * _I(self.last_action)
        self.stm_weights += (
            self.stm_learning_rate * (reward - self.stm_weights) * _I(self.last_action)
        )
        # self.stm_weights[self.stm_weights > 5] = 5

        # self.ltm_weights += self.ltm_learning_rate * reward * _I(self.last_action)
        self.ltm_weights += (
            self.ltm_learning_rate * (reward - self.ltm_weights) * _I(self.last_action)
        )

        # Update the action probabilities
        self.s_action_probs = torch.nn.functional.normalize(
            torch.exp(self.beta_stm * self.stm_weights), dim=0, p=1
        )
        self.l_action_probs = torch.nn.functional.normalize(
            torch.exp(self.beta_ltm * self.ltm_weights), dim=0, p=1
        )
