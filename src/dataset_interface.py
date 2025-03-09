"""Module providing functions for common interface for datasets."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler, LabelEncoder

class DatasetInterface:
    """Dataset interface"""

    def __init__(self):
        self.data = None
        self.labels = None
        self.train_data = None
        self.test_data = None
        self.val_data = None
        self.train_labels = None
        self.test_labels = None
        self.val_labels = None

    def fetch_from_openml(self, dataset_name):
        """
        Fetch dataset from OpenML.
        """
        self.data, self.labels = fetch_openml(dataset_name, version=1, return_X_y=True, as_frame=True)
        return self
    
    def preprocess_data(self, convert_to_binary=True, encode_labels=True, standardize_data=True, missing_values_strategy='drop', \
            remove_correlated_features=True, add_dummy_features=True):
        """
        Preprocess the dataset.
        """

        self.handle_missing_values(strategy=missing_values_strategy)

        if convert_to_binary:
            self.convert2binary()
            
        if encode_labels:
            self.encode_labels()

        if remove_correlated_features:
            self.remove_correlated_features()

        if add_dummy_features:
            num_observations = len(self.data)
            num_features = self.data.shape[1]
            required_features = int(0.5 * num_observations) 

            if num_features < required_features:
                num_dummy_features = required_features - num_features
                print(f"Adding {num_dummy_features} dummy features to meet the requirement...")
                self.add_dummy_features(num_dummy_features=num_dummy_features)

        if standardize_data:
            self.standardize_data()

        return self

    def handle_missing_values(self, strategy='drop', default_value=None):
        """
        Handle missing values in the dataset.
        """
        if strategy == 'mean':
            self.data = self.data.fillna(self.data.mean())
        elif strategy == 'median':
            self.data = self.data.fillna(self.data.median())
        elif strategy == 'mode':
            self.data = self.data.fillna(self.data.mode().iloc[0])
        elif strategy == 'default':
            if default_value is not None:
                self.data = self.data.fillna(default_value)
            else:
                raise ValueError("Default value must be provided for 'default' strategy.")
        elif strategy == 'drop':
            null_indices = self.data[self.data.isnull().any(axis=1)].index
            self.data = self.data.drop(null_indices)
            self.labels = self.labels.drop(null_indices)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        return self

    def remove_correlated_features(self, threshold=0.95):
        """
        Remove correlated features based on a given threshold.
        """
        corr_matrix = self.data.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
        self.data = self.data.drop(columns=to_drop)
        print(f"Removed {len(to_drop)} correlated features.")
        return self

    def add_dummy_features(self, num_dummy_features=10):
        """
        Add dummy features by permuting existing features.
        """
        for i in range(num_dummy_features):
            dummy_feature = self.data.iloc[:, i % self.data.shape[1]] \
                .sample(frac=1).reset_index(drop=True)
            self.data[f'dummy_{i}'] = dummy_feature
        return self

    def standardize_data(self):
        """
        Standardize the dataset (mean=0, variance=1).
        """
        scaler = StandardScaler()
        self.data = pd.DataFrame(scaler.fit_transform(self.data), columns=self.data.columns)
        return self

    def convert2binary(self):
        """
        Convert dataset labels to binary values.
        """
        if len(np.unique(self.labels)) > 2:
            self.labels = (self.labels == self.labels[0]).astype(int)  

    def encode_labels(self):
        """
        Encode the dataset labels to numeric values
        """
        label_encoder = LabelEncoder()
        self.labels = label_encoder.fit_transform(self.labels)

    def split_data(self, test_size=0.2, val_size=0.1, random_state=42):
        """
        Split data into train, test, and validation sets.
        """
        self.train_data, self.test_data, self.train_labels, self.test_labels = train_test_split(
            self.data, self.labels, test_size=test_size, random_state=random_state
        )
        self.train_data, self.val_data, self.train_labels, self.val_labels = train_test_split(
            self.train_data, self.train_labels, test_size=val_size, random_state=random_state
        )
        return self

    def get_data(self):
        """
        Return the processed dataset and labels.
        """
        return {
            'train_data': self.train_data,
            'test_data': self.test_data,
            'val_data': self.val_data,
            'train_labels': self.train_labels,
            'test_labels': self.test_labels,
            'val_labels': self.val_labels
        }
