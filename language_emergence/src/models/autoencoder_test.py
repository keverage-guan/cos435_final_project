import torch
from models.autoencoder import Autoencoder
from models.autoencoder import trainSAE
from utils.data_loading import QMatrixData

K = 5.0
kappa = 0.0
lr = 0.0
epochs = 0.0
in_channels = 0.0 
num_channels = 0.0 
filter_dim = 0.0 
kernel_dim = 0.0
maze_dim = 0.0
batch_size = 10
device = None


dataset = QMatrixData(
    q_file_path="data/teacher/q matrix dictionaries/q_matricestraining_4x4.pkl",
    label_file_path="data/teacher/label dictionaries/q_matrices_labelstraining.pkl"
)

autoencoder = Autoencoder(in_channels, maze_dim, num_channels, filter_dim, kernel_dim, K)
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)

total_losses, recon_losses, sparsity_losses, all_messages = trainSAE(autoencoder, dataset, kappa, optimizer, batch_size, epochs, device)


