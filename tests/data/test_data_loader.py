"""Test suite for the DataLoader factory."""

import pandas as pd
import numpy as np
from src.data.data_loader import OpenMLDataLoader, SyntheticDataLoader
from src.data.dataset_interface import DataInterface, DatasetInterface

def test_openml_data_loader():
    """Test fetching data from OpenML."""
    data_interface = DataInterface(OpenMLDataLoader(dataset_name='iris'))

    assert data_interface.data.data is not None
    assert data_interface.data.labels is not None
    assert isinstance(data_interface.data, DatasetInterface)
    assert isinstance(data_interface.data.data, pd.DataFrame)
    assert isinstance(data_interface.data.labels, pd.Series)

def test_synthetic_data_loader():
    """Test synthetic data generation."""
    p, n, d, g = 0.5, 1000, 10, 0.5
    data_interface = DataInterface(SyntheticDataLoader(p, n, d, g))

    assert data_interface.data.data.shape == (n, d)
    assert len(data_interface.data.labels) == n
    assert set(data_interface.data.labels).issubset({0, 1})
    mean_class_0 = data_interface.data.data[data_interface.data.labels == 0].mean()
    mean_class_1 = data_interface.data.data[data_interface.data.labels == 1].mean()
    assert np.allclose(mean_class_0, np.zeros(d), atol=0.1)
    expected_mean_class_1 = np.array([1 / (k + 1) for k in range(d)])
    assert np.allclose(mean_class_1, expected_mean_class_1, atol=0.1)
