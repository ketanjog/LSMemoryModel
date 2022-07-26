"""
Create a reward structure over context-action pairs according 
to an input object, other options are default, or random
"""
import torch

PRELOADED_CONTEXTS = {"80_20": torch.Tensor([[1, 0.80], [1, 0.2]])}


def get_loyd_context(
    num_actions: int,
    action_reward_distribution: torch.Tensor = None,
    _preload: str = None,
):
    """
    num_actions: (int) number of actions available to the agent

    action_reward_distribution: (torch.Tensor) a matrix
                            of shape |A| x 2, where each action
                            has a tuple describing the mean reward and the
                            probability of recieving it. New contexts
                            will scramble this structure over actions

    _preload: (str) is set to None by default. This option indicates whether
                            to load a predetermined reward structure
    """
    if _preload in PRELOADED_CONTEXTS.keys():
        return PRELOADED_CONTEXTS[_preload]

    elif action_reward_distribution is not None:
        return action_reward_distribution

    else:
        print("Unsure of how to proceed")
        raise NotImplementedError
