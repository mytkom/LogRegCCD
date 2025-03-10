# pylint: disable=too-few-public-methods
"""Data loader factory."""

from abc import ABC, abstractmethod
from sklearn.datasets import fetch_openml
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

class DataLoader(ABC):
    """Abstract class for loading data."""

    @abstractmethod
    def load_data(self):
        """Abstract method for data loading."""


class OpenMLDataLoader(DataLoader):
    """OpenML data creator."""

    def __init__(self, *, dataset_name, version=1):
        self.dataset_name = dataset_name
        self.version = version

    def load_data(self):
        """Fetch dataset from OpenML."""
        return fetch_openml(
            self.dataset_name, version=self.version, return_X_y=True, as_frame=True
        )


class SyntheticDataLoader(DataLoader):
    """Synthetic data creator."""

    def __init__(self, p, n, d, g):
        self.p = p
        self.n = n
        self.d = d
        self.g = g

    # TODO check - nie jestem pewna czy to jest dobrze, dodałam jako przykład factory 
    def load_data(self):
        """Generate synthetic data."""

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
            data[i, :] = multivariate_normal.rvs(mean=mean, cov=cov_matrix)

        data = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(self.d)])
        return data, labels
