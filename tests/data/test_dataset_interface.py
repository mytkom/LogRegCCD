"""Test suite for the DataInterface class."""

import pytest
import pandas as pd
import numpy as np
from pandas import testing as tm
from src.data.dataset_interface import DataInterface

def test_handle_missing_values_mean():
    """Test handling missing values with the 'mean' strategy."""
    data_interface = DataInterface()
    data_interface.data.data = pd.DataFrame({
        'feature1': [1, 2, np.nan, 4],
        'feature2': [5, np.nan, 7, 8]
    })
    data_interface.data.labels = pd.Series([0, 1, 0, 1])

    data_interface.handle_missing_values(strategy='mean')

    assert data_interface.data.data.isnull().sum().sum() == 0
    assert data_interface.data.data.iloc[2, 0] == pytest.approx(2.333, 0.001)
    assert data_interface.data.data.iloc[1, 1] == pytest.approx(6.666, 0.001)

def test_handle_missing_values_drop_observations():
    """Test handling missing values with the 'drop observations' strategy."""
    data_interface = DataInterface()
    data_interface.data.data = pd.DataFrame({
        'feature1': [1, 2, np.nan, 4],
        'feature2': [5, np.nan, 7, 8],
        'feature3': [9, np.nan, 11, 12],
    })
    data_interface.data.labels = pd.Series([0, 1, 0, 1])

    data_interface.handle_missing_values(strategy='drop observations')

    assert data_interface.data.data.shape[0] == 2
    assert data_interface.data.labels.shape[0] == 2
    assert 'feature1' in data_interface.data.data.columns
    assert 'feature2' in data_interface.data.data.columns
    assert 'feature3' in data_interface.data.data.columns
    tm.assert_frame_equal(
        data_interface.data.data,
        pd.DataFrame({
            'feature1': [1., 4.],
            'feature2': [5., 8.],
            'feature3': [9., 12.],
        })
    )

def test_handle_missing_values_drop_features():
    """Test handling missing values with the 'drop features' strategy."""
    data_interface = DataInterface()
    data_interface.data.data = pd.DataFrame({
        'feature1': [1, 2, np.nan, 4],
        'feature2': [5, np.nan, np.nan, 8],
        'feature3': [9, 10, 11, 12],
    })
    data_interface.data.labels = pd.Series([0, 1, 0, 1])

    data_interface.handle_missing_values(strategy='drop features')

    assert data_interface.data.data.shape[0] == 4
    assert data_interface.data.labels.shape[0] == 4
    assert 'feature3' in data_interface.data.data.columns
    tm.assert_frame_equal(
        data_interface.data.data,
        pd.DataFrame({
            'feature3': [9, 10, 11, 12],
        })
    )

def test_convert2binary_default():
    """Test converting labels to binary values with the 'default' strategy."""
    data_interface = DataInterface()
    data_interface.data.labels = pd.Series([3, 0, 1, 2, 0])

    data_interface.convert2binary()

    assert set(data_interface.data.labels.unique()) == {0, 1}
    tm.assert_series_equal(data_interface.data.labels, pd.Series([0, 1, 0, 0, 1]))

def test_convert2binary_most_common():
    """Test converting labels to binary values with the 'most common' strategy."""
    data_interface = DataInterface()
    data_interface.data.data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5, 6],
        'feature2': [7, 8, 9, 10, 11, 12],
    })
    data_interface.data.labels = pd.Series([2, 1, 2, 0, 2, 0])

    data_interface.convert2binary(strategy='most common')

    assert set(data_interface.data.labels.unique()) == {0, 1}
    tm.assert_series_equal(data_interface.data.labels, pd.Series([0, 0, 1, 0, 1]))
    tm.assert_frame_equal(
        data_interface.data.data,
        pd.DataFrame({
            'feature1': [1, 3, 4, 5, 6],
            'feature2': [7, 9, 10, 11, 12],
        })
    )

def test_convert2binary_chosen():
    """Test converting labels to binary values with the 'chosen' strategy."""
    data_interface = DataInterface()
    data_interface.data.data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5, 6],
        'feature2': [7, 8, 9, 10, 11, 12],
    })
    data_interface.data.labels = pd.Series([2, 1, 2, 0, 2, 0])

    data_interface.convert2binary(strategy='chosen')

    assert set(data_interface.data.labels.unique()) == {0, 1}
    tm.assert_series_equal(data_interface.data.labels, pd.Series([1, 0, 0]))
    tm.assert_frame_equal(
        data_interface.data.data,
        pd.DataFrame({
            'feature1': [2, 4, 6],
            'feature2': [8, 10, 12],
        })
    )

def test_encode_labels():
    """Test encoding labels to numeric values."""
    data_interface = DataInterface()
    data_interface.data.labels = pd.Series(['cat', 'dog', 'cat', 'bird'])

    data_interface.encode_labels()

    assert len(np.unique(data_interface.data.labels)) == 3
    assert isinstance(data_interface.data.labels[0], np.int64)
    tm.assert_series_equal(data_interface.data.labels, pd.Series([1, 2, 1, 0]))

def test_remove_correlated_features():
    """Test removing correlated features."""
    data_interface = DataInterface()
    data_interface.data.data = pd.DataFrame({
        'feature1': [1, 2, 3, 4],
        'feature2': [1, 2, 3, 4],
        'feature3': [0, 6, 7, 8],
        'feature4': [3, 3, 3, 4],
        'feature5': [6, 6, 6, 8]
    })

    data_interface.remove_correlated_features(threshold=0.95)
    assert 'feature2' not in data_interface.data.data.columns
    assert 'feature5' not in data_interface.data.data.columns
    assert len(data_interface.data.data.columns) == 3

def test_add_dummy_features():
    """Test adding dummy features."""
    data_interface = DataInterface()
    data_interface.data.data = pd.DataFrame({
        'feature1': [1, 2, 3, 4]
    })

    data_interface.add_dummy_features(num_dummy_features=2)

    assert 'dummy_0' in data_interface.data.data.columns
    assert 'dummy_1' in data_interface.data.data.columns
    assert len(data_interface.data.data.columns) == 3

def test_standardize_data():
    """Test standardizing the dataset."""
    data_interface = DataInterface()
    data_interface.data.data = pd.DataFrame({
        'feature1': [1, 2, 3, 4],
        'feature2': [5, 6, 7, 8]
    })

    data_interface.standardize_data()

    print(f'Mean of each column:\n{data_interface.data.data.mean()}')
    print(f'Standard deviation of each column:\n{data_interface.data.data.std()}')

    assert np.allclose(data_interface.data.data.mean(), 0, atol=1e-7)
    assert np.allclose(data_interface.data.data.std(ddof=0), 1, atol=1e-7)

def test_split_data():
    """Test splitting data into train, test, and validation sets."""
    data_interface = DataInterface()
    data_interface.data.data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'feature2': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    })
    data_interface.data.labels = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    data_interface.split_data(test_size=0.2, val_size=0.25)

    assert len(data_interface.train_data.data) == 6
    assert len(data_interface.test_data.data) == 2
    assert len(data_interface.val_data.data) == 2
    assert len(data_interface.train_data.labels) == 6
    assert len(data_interface.test_data.labels) == 2
    assert len(data_interface.val_data.labels) == 2

def test_get_data():
    """Test retrieving processed data."""
    data_interface = DataInterface()
    data_interface.data.data = pd.DataFrame({
        'feature1': [1, 2, 3, 4],
        'feature2': [5, 6, 7, 8]
    })
    data_interface.data.labels = pd.Series([0, 1, 0, 1])

    data = data_interface.get_data()

    assert isinstance(data, dict)
    assert 'train_data' in data
    assert 'test_data' in data
    assert 'val_data' in data
    assert 'train_labels' in data
    assert 'test_labels' in data
    assert 'val_labels' in data
