"""Module providing functions for Exploratory Data Analysis."""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

def figure_missing_values(dataset, figsize=(10, 6)):
    """Generate a heatmap of missing values in the dataset."""
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(dataset.data.isnull(), cbar=False, cmap='viridis', ax=ax)
    ax.set_title("Missing Values Heatmap")
    return fig

def figure_target_distribution(dataset, figsize=(8, 6)):
    """Generate the distribution of the target variable."""
    fig, ax = plt.subplots(figsize=figsize)
    sns.countplot(x=dataset.labels, ax=ax)
    ax.set_title("Target Variable Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    return fig

def figure_feature_distributions(dataset, figsize=(15, 15)):
    """Generate distributions of all features."""
    fig = plt.figure(figsize=figsize)
    dataset.data.hist(bins=30, figsize=figsize)
    plt.suptitle("Feature Distributions")
    return fig

def figure_correlation_matrix(dataset, figsize=(12, 10)):
    """Generate a correlation matrix for the features."""
    fig, ax = plt.subplots(figsize=figsize)
    corr_matrix = dataset.data.corr()
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
    ax.set_title("Feature Correlation Matrix")
    return fig

def figure_pairplot(dataset, sample_size=100):
    """Generate pairwise relationships between features (sampled for performance)."""
    sample_data = dataset.data.sample(sample_size)
    sample_data['label'] = dataset.labels[sample_data.index]
    g = sns.pairplot(sample_data, hue='label', palette='viridis')
    g.fig.suptitle("Pairwise Feature Relationships", y=1.02)
    return g.fig

def figure_tsne(dataset, perplexity=30, random_state=42, figsize=(10, 8)):
    """Generate t-SNE visualization of the dataset."""
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    tsne_results = tsne.fit_transform(dataset.data)
    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=dataset.labels, palette='viridis', ax=ax)
    ax.set_title("t-SNE Visualization")
    return fig
