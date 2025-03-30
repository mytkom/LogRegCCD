# pylint: disable=too-few-public-methods
"""Data loader factory."""

from abc import ABC, abstractmethod
from sklearn.datasets import fetch_openml
import numpy as np
import pandas as pd


class DataLoader(ABC):
    """Abstract class for loading data."""

    @abstractmethod
    def load_data(self):
        """Abstract method for data loading."""


class OpenMLDataLoader(DataLoader):
    """OpenML data creator."""

    def __init__(self, dataset_name, version=1):
        self.dataset_name = dataset_name
        self.version = version

    def load_data(self):
        """Fetch dataset from OpenML."""
        return fetch_openml(
            self.dataset_name, version=self.version, return_X_y=True, as_frame=True
        )


class SyntheticDataLoader(DataLoader):
    """Synthetic data creator."""

    def __init__(self, p, n, d, g, random_seed=42): # pylint: disable=too-many-arguments, too-many-positional-arguments
        """
        Parameters:
        - p: Class prior probability for Y=1.
        - n: Number of observations.
        - d: Number of features.
        - g: Covariance parameter for the multivariate normal distribution.
        - random_seed: Random seed.
        """
        self.p = p
        self.n = n
        self.d = d
        self.g = g
        self.random_seed = random_seed

    # pylint: disable=fixme
    # TODO check - nie jestem pewna czy to jest dobrze, dodałam jako przykład factory
    def load_data(self):
        """Generate synthetic data."""
        np.random.seed(self.random_seed)

        labels = np.random.binomial(1, self.p, self.n)

        cov_matrix = np.zeros((self.d, self.d))
        for i in range(self.d):
            for j in range(self.d):
                cov_matrix[i, j] = self.g ** abs(i - j)

        data = np.zeros((self.n, self.d))
        for i in range(self.n):
            if labels[i] == 0:
                mean = np.zeros(self.d)
            else:
                mean = np.array([1 / (k + 1) for k in range(self.d)])
            data[i, :] = np.random.multivariate_normal(mean=mean, cov=cov_matrix)

        data = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(self.d)])
        return data, labels
