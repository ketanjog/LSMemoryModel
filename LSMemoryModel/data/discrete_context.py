import numpy as np
import torch

from LSMemoryModel.constants.discrete import num_actions, num_contexts

optimal_actions = np.random.permutation(num_actions)[:num_contexts]
contexts = np.arange(num_contexts)

DISCRETE_CONTEXT = (torch.tensor(contexts), torch.tensor(optimal_actions))
