import torch
import numpy as np
from LSMemoryModel.constants.discrete_context import num_contexts, num_actions

optimal_actions = np.random.permutation(num_actions)[:num_contexts]
contexts = np.arange(num_contexts)

DISCRETE_CONTEXT = (torch.tensor(contexts), torch.tensor(optimal_actions))