# pca, t-sne, umap, and loss visualizers

import os
import re
import matplotlib.pyplot as plt

_FIG_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'outputs', 'figures')

def _save(filename):
    os.makedirs(_FIG_DIR, exist_ok=True)
    plt.savefig(os.path.join(_FIG_DIR, filename), bbox_inches='tight', dpi=150)

def plot_pca_variance(pca, save_name='pca_variance.png'):
    plt.figure()
    plt.bar(range(1, 6), pca.explained_variance_ratio_ * 100)
    plt.xlabel('PC index')
    plt.ylabel('Variance explained (%)')
    plt.ylim(0, 100)
    _save(save_name)
    plt.show()

def plot_pca_by_label(pca_result, labels, title):
    plt.figure()
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=labels, cmap='tab20')
    plt.xlabel('first PC')
    plt.ylabel('second PC')
    plt.colorbar()
    plt.title(title)
    safe = re.sub(r'[^\w\-]', '_', title)[:60]
    _save(f'pca_{safe}.png')
    plt.show()
