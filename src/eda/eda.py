"""Module providing functions for Exploratory Data Analysis."""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

def plot_missing_values(dataset):
    """Plot a heatmap of missing values in the dataset."""
    plt.figure(figsize=(10, 6))
    sns.heatmap(dataset.data.isnull(), cbar=False, cmap='viridis')
    plt.title("Missing Values Heatmap")
    plt.show()

def plot_target_distribution(dataset):
    """Plot the distribution of the target variable."""
    plt.figure(figsize=(8, 6))
    sns.countplot(x=dataset.labels)
    plt.title("Target Variable Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.show()

def plot_feature_distributions(dataset):
    """Plot distributions of all features."""
    dataset.data.hist(bins=30, figsize=(15, 15))
    plt.suptitle("Feature Distributions")
    plt.show()

def plot_correlation_matrix(dataset):
    """Plot a correlation matrix for the features."""
    plt.figure(figsize=(12, 10))
    corr_matrix = dataset.data.corr()
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Feature Correlation Matrix")
    plt.show()

def plot_pairplot(dataset, sample_size=100):
    """Plot pairwise relationships between features (sampled for performance)."""
    sample_data = dataset.data.sample(sample_size)
    sample_data['label'] = dataset.labels[sample_data.index]
    sns.pairplot(sample_data, hue='label', palette='viridis')
    plt.suptitle("Pairwise Feature Relationships", y=1.02)
    plt.show()

def plot_tsne(dataset, perplexity=30):
    """Plot t-SNE visualization of the dataset."""
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    tsne_results = tsne.fit_transform(dataset.data)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], \
        hue=dataset.labels, palette='viridis')
    plt.title("t-SNE Visualization")
    plt.show()
