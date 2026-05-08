# pca, t-sne, umap, and loss visualizers

import os
import re
import matplotlib.pyplot as plt
import numpy as np

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
    plt.figure() # Back to your original default size
    
    # Reverting to your exact scatter style
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=labels, cmap='tab20')
    
    plt.xlabel('first PC')
    plt.ylabel('second PC')
    plt.title(title)

    # Replace colorbar with discrete legend
    handles, _ = scatter.legend_elements()
    
    # Mapping for arrow symbols
    action_map = {0: "↑ Up", 1: "↓ Down", 2: "← Left", 3: "→ Right"}
    unique_labels = np.unique(labels)

    if set(unique_labels).issubset({0, 1, 2, 3}) and "Action" in title:
        legend_labels = [action_map[int(l)] for l in unique_labels]
    else:
        legend_labels = [f"Pos {int(l)}" for l in unique_labels]

    plt.legend(handles, legend_labels, title="Legend", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Your original save logic
    safe = re.sub(r'[^\w\-]', '_', title)[:60]
    _save(f'pca_{safe}.png')
    
    plt.show()