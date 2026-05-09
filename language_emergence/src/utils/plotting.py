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

def plot_pca_1d_by_label(pca_result, labels, title, jitter=0.15, seed=0):
    """
    1-D strip plot along PC1, with points organised into per-label y-lanes
    (plus a small within-lane jitter to separate overlapping points).
    """
    rng = np.random.default_rng(seed)
    pc1 = pca_result[:, 0]

    unique_labels = np.unique(labels)
    n_labels = len(unique_labels)
    cmap = plt.get_cmap('tab20')
    color_map = {l: cmap(i / max(n_labels - 1, 1)) for i, l in enumerate(unique_labels)}

    # Assign each label a fixed y-centre, evenly spaced
    y_centers = {l: i for i, l in enumerate(unique_labels)}

    fig, ax = plt.subplots(figsize=(10, max(3, n_labels * 0.6)))

    for l in unique_labels:
        mask = np.array(labels) == l
        n_pts = mask.sum()
        y_vals = y_centers[l] + rng.uniform(-jitter, jitter, size=n_pts)

        label_str = action_map[int(l)] if set(unique_labels).issubset({0,1,2,3}) and "Action" in title \
                    else str(int(l)) if isinstance(l, (int, np.integer, float)) else str(l)

        ax.scatter(pc1[mask], y_vals, c=[color_map[l]], s=18, alpha=0.7,
                   label=label_str, linewidths=0)

    # Draw a subtle guide line at each label's y-centre
    for l in unique_labels:
        ax.axhline(y_centers[l], color='grey', linewidth=0.4, linestyle='--', zorder=0)

    ax.set_xlabel('First PC')
    ax.set_yticks(list(y_centers.values()))
    ax.set_yticklabels([
        action_map[int(l)] if set(unique_labels).issubset({0,1,2,3}) and "Action" in title
        else str(int(l)) if isinstance(l, (int, np.integer, float)) else str(l)
        for l in unique_labels
    ])
    ax.set_title(title)
    ax.legend(title="Legend", bbox_to_anchor=(1.05, 1), loc='upper left')

    safe = re.sub(r'[^\w\-]', '_', title)[:60]
    _save(f'pca_1d_{safe}.png')
    plt.show()