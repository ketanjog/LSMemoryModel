"""
Set of evaluation functions
"""

import torch
from torch.distributions.categorical import Categorical


def softmax(values: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """
    returns a softmax probability distribution
    based on the values vector, where each term is
    weighted by the beta parameter
    """

    probabilities: torch.Tensor = torch.nn.functional.normalize(
        torch.exp(beta * values), dim=0, p=1
    )

    return probabilities


def sample(distribution: torch.Tensor) -> int:
    """
    Given a probability distribution over indices,
    returns an index value
    """
    choice: int = Categorical(distribution).sample()

    return choice


def onehot(index: int, size: int) -> torch.Tensor:
    """
    Given an index and total number of options,
    returns a one-hot vector representation of the
    given index
    """
    onehot: torch.Tensor = torch.zeros(size)
    onehot[index] = 1

    return onehot
