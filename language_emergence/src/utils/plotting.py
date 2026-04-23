# pca, t-sne, umap, and loss visualizers

import matplotlib.pyplot as plt

def plot_pca_variance(pca):

    plt.figure()
    plt.bar(range(1, 6), pca.explained_variance_ratio_ * 100)
    plt.xlabel('PC index')
    plt.ylabel('Variance explained (%)')
    plt.ylim(0, 100)
    plt.savefig('variance.png')

def plot_pca_by_label(pca_result, labels, title):

    plt.figure()
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=labels, cmap='tab20')
    plt.xlabel('first PC')
    plt.ylabel('second PC')
    plt.colorbar()
    plt.title(title)
