"""Module providing functions for Exploratory Data Analysis."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import missingno as msno
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA


def print_basic_info(dataset):
    """
    Print basic info about dataset.
    """
    print("\nDataset shape:")
    print(dataset.data.shape)

    print("\nDataset features count:")
    print(dataset.data.shape[1])

    print("\nDataset observation count:")
    print(dataset.data.shape[0])

    print("\nDataset overview:")
    print(dataset.data.head())

    print("\nDataset description:")
    print(dataset.data.describe())

    print("\nDataset missing values count:")
    print(dataset.data.isnull().sum().sum())

    constant_features = [col for col in dataset.data.columns if dataset.data[col].nunique() == 1]
    print("\nConstant features count:")
    print(len(constant_features))

    print("\nLabels description:")
    print(dataset.labels.describe())

    print("\nLabels missing values count:")
    print(dataset.labels.isnull().sum().sum())


def print_target_distribution_info(dataset):
    """
    Print info about distribution of labels.
    """
    class_distribution = dataset.labels.value_counts()
    class_percentage = dataset.labels.value_counts(normalize=True) * 100

    class_distribution_df = pd.DataFrame({
        'Count': class_distribution,
        'Percentage': class_percentage
    })

    print(class_distribution_df)


def figure_missing_values(dataset, figsize=(10, 6)):
    """Generate a heatmap of missing values in the dataset."""
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(dataset.data.isnull(), cbar=False, cmap='viridis', ax=ax)
    ax.set_title("Missing Values Heatmap")
    return fig


def figure_msno_matrix(dataset, num_features=None, figsize=(12, 10)):
    """Generate a missingno matrix to visualize missing values in the dataset."""

    if num_features is not None:
        selected_features = np.random.choice(dataset.data.columns, num_features, replace=False)
        dataset = dataset.data[selected_features]
    else:
        dataset = dataset.data

    fig, ax = plt.subplots(figsize=figsize)
    msno.matrix(dataset, ax=ax)
    ax.set_title("Missingno Matrix")

    return fig

def figure_target_distribution(dataset, figsize=(8, 6)):
    """Generate the distribution of the target variable."""
    fig, ax = plt.subplots(figsize=figsize)
    sns.countplot(x=dataset.labels, ax=ax)
    ax.set_title("Target Variable Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    return fig


def figure_feature_distributions(dataset, num_features=5, figsize=(15, 15)):
    """Generate distributions of a random subset of features."""

    if num_features is not None:
        selected_features = np.random.choice(dataset.data.columns, num_features, replace=False)
        dataset = dataset.data[selected_features]
    else:
        dataset = dataset.data

    fig = plt.figure(figsize=figsize)
    dataset.hist(bins=30, figsize=figsize)
    plt.suptitle("Feature Distributions")
    return fig


def figure_correlation_matrix(dataset, num_features=None, figsize=(12, 10)):
    """Generate a correlation matrix for a random subset of features."""

    if num_features is not None:
        selected_features = np.random.choice(dataset.data.columns, num_features, replace=False)
        dataset = dataset.data[selected_features]
    else:
        dataset = dataset.data

    fig, ax = plt.subplots(figsize=figsize)
    corr_matrix = dataset.corr()
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
    ax.set_title("Feature Correlation Matrix")

    return fig


def figure_pairplot(dataset, sample_size=100, figsize=(15, 15)):
    """Generate pairwise relationships between features (sampled for performance)."""
    sample_data = dataset.data.sample(sample_size)
    sample_data['label'] = dataset.labels[sample_data.index]
    fig = plt.figure(figsize=figsize)
    sns.pairplot(sample_data, hue='label', palette='viridis')
    plt.suptitle("Pairwise Feature Relationships", y=1.02)

    return fig


def figure_tsne(dataset, perplexity=30, random_state=42, figsize=(10, 8)):
    """Generate t-SNE visualization of the dataset."""
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    tsne_results = tsne.fit_transform(dataset.data)
    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(
        x=tsne_results[:, 0],
        y=tsne_results[:, 1],
        hue=dataset.labels,
        palette='viridis', ax=ax
    )
    ax.set_title("t-SNE Visualization")
    return fig


def feature_variance_explained(dataset, n_components=None):
    """
    Determmines how the features explain the variance.
    """
    pca = PCA(n_components=n_components)
    pca.fit(dataset.data)

    variance_explained = pca.explained_variance_ratio_
    components = pca.components_

    feature_importance = pd.DataFrame(
        abs(components.T) @ variance_explained,
        index=dataset.data.columns,
        columns=["Variance Explained"]
    )

    return feature_importance.sort_values(by="Variance Explained", ascending=False)


def train_logistic_regression(
        dataset, regularization_type=None, c=1.0, solver='liblinear', max_iter=500, verbosity=2
):
    """
    Test logistic regression performance for a given dataset.
    """

    if regularization_type == 'l1':
        model = LogisticRegression(penalty='l1', solver=solver, C=c, max_iter=max_iter)
    else:
        model = LogisticRegression(solver=solver, max_iter=max_iter)

    model.fit(dataset.train_data.data, dataset.train_data.labels)
    y_pred = model.predict(dataset.test_data.data)
    accuracy = accuracy_score(dataset.test_data.labels, y_pred)

    if verbosity > 0:
        print(f"{regularization_type if regularization_type else 'No regularization'}:")
        print(f'Accuracy: {accuracy}')

    if verbosity > 1:
        print("Classification report:")
        print(classification_report(dataset.test_data.labels, y_pred))
        print("Confusion matrix:")
        print(confusion_matrix(dataset.test_data.labels, y_pred))

    return model, y_pred, accuracy
