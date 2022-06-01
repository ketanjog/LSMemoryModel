from LSMemoryModel.algos.base_algo import BaseAlgo
import torch
from torch.distributions.categorical import Categorical

class PolicyGradient(BaseAlgo):
    def __init__(self, stm_learning_rate, num_actions, beta, run_id=None):
        super().__init__(run_id)
        self.name = "policy_gradient"
        self.num_actions = num_actions
        # Short Term Memory Weights (Try random seeding)
        self.stm_weights = torch.ones(self.num_actions)
        # self.stm_weights /= torch.sum(self.stm_weights)

        # Probabilities of taking each action
        self.action_probs = torch.ones(self.num_actions)
        self.action_probs = torch.nn.functional.normalize(self.action_probs,dim=0,p=1)


        # STM Learning Rate        
        self.stm_learning_rate = stm_learning_rate


        self.last_action = None

        self.beta = beta


    
    def choose_action(self, context):
        """
        Chooses the action with the maximum weight.
        If there are multiple max actions, a max action is
        randomly sampled
        """
        try:
            self.last_action = Categorical(self.action_probs).sample()
        except: 
            print(self.stm_weights)
            # This is where overflow occurs, so we choose the previous action
            # Which is the one this policy would have chosen anyway, had we evaluated
            # the exponential
            # pass


        return self.last_action

    def update(self, reward):
        """
        Updates the weight of the chosen action
        using the exponential learning rule.
        """

        def _I(a: int) -> int:
            indicator = torch.zeros(self.num_actions)
            # indicator = torch.full((num_actions,), prob_epsilon)
            indicator[a] = 1
            return indicator

        # Update the value function

        # Tried Reciprocal: (1 / torch.maximum((_I(self.last_action) - self.action_probs), torch.full((num_actions,), prob_epsilon)))
        # self.stm_weights += self.stm_learning_rate * reward * _I(self.last_action)
        # self.stm_weights[self.stm_weights > 5] = 5
        self.stm_weights += self.stm_learning_rate * (torch.full((self.num_actions,), reward/2 + 0.5) - self.action_probs) * _I(self.last_action)

        # self.stm_weights += self.stm_learning_rate * reward * (_I(self.last_action) - self.action_probs)
        # self.stm_weights += self.stm_learning_rate * reward * (_I(self.last_action) - torch.full((num_actions,), (1-reward)/2) + reward * self.action_probs)
        # self.stm_weights = torch.nn.functional.normalize(self.stm_weights, dim=0,p=2)
        # self.print = (_I(self.last_action) - self.action_probs)[self.last_action]
        # Update the action probabilities
        self.action_probs = torch.nn.functional.normalize(torch.exp(self.beta * self.stm_weights), dim=0,p=1)

        # print("STM W: " + str(self.stm_weights))
        # print("--------------------------")
        # print("ACTION PR: " + str(self.action_probs))

class DualPolicyGradient(PolicyGradient):
    def __init__(self, stm_learning_rate, ltm_learning_rate, num_actions, run_id=None):
        super().__init__(stm_learning_rate, num_actions, run_id)
        self.name = "policy_gradient"

        # LTM Weights (Try random seeding)
        self.ltm_weights = torch.ones(self.num_actions)
        # self.stm_weights /= torch.sum(self.stm_weights)

        # Probabilities of taking each action
        self.s_action_probs = torch.ones(self.num_actions)
        self.s_action_probs = torch.nn.functional.normalize(self.s_action_probs,dim=0,p=1)

        self.l_action_probs = torch.ones(self.num_actions)
        self.l_action_probs = torch.nn.functional.normalize(self.l_action_probs,dim=0,p=1)

        self.action_probs = torch.ones(self.num_actions)
        self.action_probs = torch.nn.functional.normalize(self.action_probs,dim=0,p=1)


        # STM Learning Rate        
        self.ltm_learning_rate = ltm_learning_rate
        self.stm_learning_rate = stm_learning_rate


        self.last_action = None

        self.beta = 10


    
    def choose_action(self, context):
        """
        Chooses the action with the maximum weight.
        If there are multiple max actions, a max action is
        randomly sampled
        """
        self.action_probs = torch.nn.functional.normalize(self.s_action_probs*self.l_action_probs,dim=0,p=1)

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
        self.stm_weights += self.stm_learning_rate * reward * _I(self.last_action)
        self.stm_weights[self.stm_weights > 5] = 5

        self.ltm_weights += self.ltm_learning_rate * reward * _I(self.last_action)

        # Update the action probabilities
        self.s_action_probs = torch.nn.functional.normalize(torch.exp(self.stm_weights), dim=0,p=1)
        self.l_action_probs = torch.nn.functional.normalize(torch.exp(self.beta*self.ltm_weights), dim=0,p=1)

