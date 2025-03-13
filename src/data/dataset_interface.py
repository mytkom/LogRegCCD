"""Module providing functions for common interface for datasets."""

import dataclasses
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


@dataclasses.dataclass
class DatasetInterface:
    """Class to encapsulate data and labels."""

    def __init__(self):
        self.data = None
        self.labels = None


class DataInterface:
    """Dataset interface"""

    def __init__(self, data_loader=None):
        self.data = DatasetInterface()
        if data_loader:
            self.data.data, self.data.labels = data_loader.load_data()
        self.train_data = DatasetInterface()
        self.val_data = DatasetInterface()
        self.test_data = DatasetInterface()

    def preprocess_data(self, missing_values_strategy='mean', ratio=0.5):
        """
        Preprocess the dataset.
        """
        self.handle_missing_values(strategy=missing_values_strategy)
        self.encode_labels()
        self.convert2binary()
        self.remove_correlated_features()

        num_observations = len(self.data.data)
        num_features = self.data.data.shape[1]
        required_features = int(ratio * num_observations)
        if num_features < required_features:
            num_dummy_features = required_features - num_features
            print(
                f"Adding {num_dummy_features} dummy features"
            )
            self.add_dummy_features(num_dummy_features=num_dummy_features)

        self.standardize_data()

        return self

    def handle_missing_values(self, strategy='drop', default_value=None, reset_index=True):
        """
        Handle missing values in the dataset.
        """
        if strategy == 'mean':
            self.data.data = self.data.data.fillna(self.data.data.mean())
        elif strategy == 'median':
            self.data.data = self.data.data.fillna(self.data.data.median())
        elif strategy == 'mode':
            self.data.data = self.data.data.fillna(self.data.data.mode().iloc[0])
        elif strategy == 'default':
            if default_value is not None:
                self.data.data = self.data.data.fillna(default_value)
            else:
                raise ValueError(
                    "Default value must be provided for 'default' strategy."
                )
        elif strategy == 'drop':
            mask = ~self.data.data.isnull().any(axis=1)
            self.data.data = self.data.data[mask]
            self.data.labels = self.data.labels[mask]
            if reset_index:
                self.data.data.reset_index(drop=True, inplace=True)
                self.data.labels.reset_index(drop=True, inplace=True)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        return self

    def remove_correlated_features(self, threshold=0.95):
        """
        Remove correlated features based on a given threshold.
        """
        corr_matrix = self.data.data.corr().abs()
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1
        ).astype(bool))
        to_drop = [
            column for column in upper_tri.columns
            if any(upper_tri[column] > threshold)
        ]
        self.data.data = self.data.data.drop(columns=to_drop)
        print(f"Removed {len(to_drop)} correlated features.")
        return self

    def add_dummy_features(self, num_dummy_features=10):
        """
        Add dummy features by permuting existing features.
        """
        for i in range(num_dummy_features):
            dummy_feature = self.data.data.iloc[:, i % self.data.data.shape[1]] \
                .sample(frac=1).reset_index(drop=True)
            self.data.data[f'dummy_{i}'] = dummy_feature
        return self

    def standardize_data(self):
        """
        Standardize the dataset (mean=0, variance=1).
        """
        scaler = StandardScaler()
        self.data.data = pd.DataFrame(
            scaler.fit_transform(self.data.data), columns=self.data.data.columns
        )
        return self

    def convert2binary(self, strategy='default', in_labels=None, reset_index=True):
        """
        Convert dataset labels to binary values.
        """
        if len(np.unique(self.data.labels)) > 2:

            if strategy == 'default':
                self.data.labels = (self.data.labels == self.data.labels[0]).astype(int)

            elif strategy in ('most common', 'chosen'):

                if strategy == 'most common':
                    label_counts = self.data.labels.value_counts()
                    in_labels = label_counts.index[:2]
                elif in_labels is None:
                    in_labels = [0, 1]

                mask = self.data.labels.isin(in_labels)
                self.data.data = self.data.data[mask]
                self.data.labels = self.data.labels[mask]

                self.data.labels = (self.data.labels != in_labels[0]).astype(int)

                if reset_index:
                    self.data.data.reset_index(drop=True, inplace=True)
                    self.data.labels.reset_index(drop=True, inplace=True)

            else:
                raise ValueError(f"Unknown strategy: {strategy}")

        return self

    def encode_labels(self):
        """
        Encode the dataset labels to numeric values.
        """
        label_encoder = LabelEncoder()
        self.data.labels = pd.Series(label_encoder.fit_transform(self.data.labels))

    def split_data(self, test_size=0.2, val_size=0.1, random_state=42):
        """
        Split data into train, test, and validation sets.
        """
        self.train_data.data, self.test_data.data, \
        self.train_data.labels, self.test_data.labels = train_test_split(
            self.data.data, self.data.labels, test_size=test_size, random_state=random_state
        )

        self.train_data.data, self.val_data.data, \
        self.train_data.labels, self.val_data.labels = train_test_split(
            self.train_data.data, self.train_data.labels, test_size=val_size, \
                random_state=random_state
        )
        return self

    def get_data(self):
        """
        Return the processed dataset and labels.
        """
        return {
            'train_data': self.train_data.data,
            'test_data': self.test_data.data,
            'val_data': self.val_data.data,
            'train_labels': self.train_data.labels,
            'test_labels': self.test_data.labels,
            'val_labels': self.val_data.labels
        }
