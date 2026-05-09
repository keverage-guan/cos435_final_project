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
    action_map = {0: "→ Right", 1: "↑ Up", 2: "← Left", 3: "↓ Down"}
    unique_labels = np.unique(labels)

    if set(unique_labels).issubset({0, 1, 2, 3}) and "Action" in title:
        legend_labels = [action_map[int(l)] for l in unique_labels]
    else:
        legend_labels = [f"{int(l)}" for l in unique_labels]

    plt.legend(handles, legend_labels, title="Legend", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Your original save logic
    safe = re.sub(r'[^\w\-]', '_', title)[:60]
    _save(f'pca_{safe}.png')
    
    plt.show()

def plot_pca_1d_by_label(pca_result, labels, title, jitter=0.25, seed=0):
    """
    1-D strip plot along PC1, with random y-jitter to separate overlapping points.
    Use when PCA variance is dominated by the first component.
    """
    rng = np.random.default_rng(seed)
    pc1 = pca_result[:, 0]

    unique_labels = np.unique(labels)
    cmap = plt.get_cmap('tab20')
    color_map = {l: cmap(i / max(len(unique_labels) - 1, 1)) for i, l in enumerate(unique_labels)}
    colors = [color_map[l] for l in labels]

    y_jitter = rng.uniform(-jitter, jitter, size=len(pc1))

    fig, ax = plt.subplots(figsize=(10, 3))
    for l in unique_labels:
        mask = np.array(labels) == l
        label_str = action_map[int(l)] if set(unique_labels).issubset({0,1,2,3}) and "Action" in title \
                    else str(int(l)) if isinstance(l, (int, np.integer, float)) else str(l)
        ax.scatter(pc1[mask], y_jitter[mask], c=[color_map[l]], s=18, alpha=0.7,
                   label=label_str, linewidths=0)

    ax.axhline(0, color='grey', linewidth=0.5, linestyle='--')
    ax.set_xlabel('First PC')
    ax.set_yticks([])
    ax.set_title(title)
    ax.legend(title="Legend", bbox_to_anchor=(1.05, 1), loc='upper left')

    safe = re.sub(r'[^\w\-]', '_', title)[:60]
    _save(f'pca_1d_{safe}.png')
    plt.show()