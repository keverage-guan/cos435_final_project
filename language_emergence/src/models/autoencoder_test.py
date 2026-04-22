import torch
import sys, os

# src folder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.autoencoder import Autoencoder
from models.autoencoder import trainSAE, testSAE
from utils.data_loading import QMatrixData

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

K = 5
kappa = 1/500
lr = 5e-4
train_epochs = 50
in_channels = 4 
num_channels = 10
kernel_dim = 2
maze_dim = 4
batch_size = 10
device = None

train_dataset = QMatrixData(
    q_file_path=os.path.join(BASE_DIR, "data/teacher/q matrix dictionaries/q_matricestraining_4x4.pkl"),
    label_file_path=os.path.join(BASE_DIR, "data/teacher/label dictionaries/q_matrices_labelstraining_4x4.pkl")
)
test_dataset = QMatrixData(
    q_file_path=os.path.join(BASE_DIR, "data/teacher/q matrix dictionaries/q_matricestest_4x4.pkl"),
    label_file_path=os.path.join(BASE_DIR, "data/teacher/label dictionaries/q_matrices_labelstest_4x4.pkl")
)

autoencoder = Autoencoder(in_channels, maze_dim, num_channels, kernel_dim, K)
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)

# training language
total_losses, recon_losses, sparsity_losses, all_messages = trainSAE(autoencoder, train_dataset, kappa, optimizer, batch_size, train_epochs, device)

# getting messages from trained language
_, _, _, all_messages = testSAE(autoencoder, test_dataset, batch_size, kappa, device)

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Stack all messages into a single tensor
messages = torch.cat(all_messages, dim=0)  # shape: (num_tasks, K)
messages_np = messages.numpy()

pca = PCA(n_components=5)
pca_result = pca.fit_transform(messages_np)

# Plot explained variance (Fig 2a(i))
plt.bar(range(1, 6), pca.explained_variance_ratio_ * 100)
plt.xlabel('PC index')
plt.ylabel('Variance explained (%)')
plt.ylim(0, 100)
plt.savefig('variance.png')

# Fig 2a(ii) - color by wall position
# plt.scatter(pca_result[:, 0], pca_result[:, 1], c=wall_labels)

# Fig 2a(iii) - color by goal location  
# plt.scatter(pca_result[:, 0], pca_result[:, 1], c=goal_labels)