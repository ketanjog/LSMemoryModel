"""
Set of evaluation functions
"""

from dis import dis
import torch
from torch.distributions.categorical import Categorical
import math


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


def gaussianKL(arr1, arr2) -> float:
    """
    Returns the KL divergence between the input arrays
    Assumes that both arrays are multivariate gaussians,
    With 0 Covariance values

    assumes both inputs are dictionaries with 2 arrays titled
    "mean" and "variance"

    """
    assert len(arr1) == len(arr2), "The arrays are not of the same length"

    KL_divergence: float = 0.0

    for i in range(len(arr1)):
        log_term = torch.log(arr2["variance"][i]) - torch.log(arr1["variance"][i])
        second_term = (
            arr1["variance"][i] ** 2 + (arr1["mean"][i] - arr2["mean"][i]) ** 2
        ) / (2 * arr2["variance"][i] ** 2)

        constant_term = 0.5

        KL_divergence += log_term + second_term - constant_term

    return KL_divergence


def sample_normal(mean, variance):
    return torch.normal(mean, math.sqrt(variance), size=(1,)).item()


def _l1(arr1: torch.Tensor, arr2: torch.Tensor) -> torch.Tensor:
    """
    Returns l1 squared distance
    """
    diff = arr1 - arr2
    l1 = torch.sum(diff) ** 2

    return l1


def _exponential_kernel(arr: torch.Tensor, radius=1) -> torch.Tensor:
    exponential = torch.exp(-arr / (2 * radius**2))

    return exponential


def rbf_l1(arr1: torch.Tensor, arr2: torch.Tensor, radius: float) -> torch.Tensor:
    assert len(arr1) == len(arr2), "Arrays are not of the same length"
    l1 = _l1(arr1, arr2)
    distance = _exponential_kernel(l1, radius=radius)

    return distance
