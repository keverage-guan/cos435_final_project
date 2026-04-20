# Convolutional Autoencoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

class Autoencoder(nn.Module):

    def __init__(self, in_channels, maze_dim, num_channels, filter_dim, kernel_dim, K):
        
        # encoder layers
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=num_channels, kernel_size=kernel_dim, padding=filter_dim-1)
        self.conv2 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=kernel_dim, padding=filter_dim-1)

        conv2_shape = Autoencoder.output_dim((num_channels, maze_dim, maze_dim), self.conv2)
        flattened_size = conv2_shape[0]*conv2_shape[1]*conv2_shape[2]

        self.flatten = nn.Flatten(num_channels, 1)
        self.linear1 = nn.Linear(flattened_size, K)
        
        # decoder layers
        self.linear2 = nn.Linear(K, num_channels, flattened_size)
        self.unflatten = nn.Unflatten(flattened_size, conv2_shape)
        self.deconv1 = nn.ConvTranspose2d(in_channels=conv2_shape[0], out_channels=num_channels, kernel_size=kernel_dim, padding=filter_dim-1)
        self.deconv1 = nn.ConvTranspose2d(in_channels=num_channels, out_channels=4, kernel_size=kernel_dim, padding=filter_dim-1)
    
    def output_dim(input_dim, conv_layer):

        h_in = input_dim[1]
        w_in = input_dim[2]

        padding = conv_layer.padding if isinstance(conv_layer.padding, tuple) else (conv_layer.padding,)
        kernel_size = conv_layer.kernel_size if isinstance(conv_layer.kernel_size, tuple) else (conv_layer.kernel_size,)
        dilation = conv_layer.dilation if isinstance(conv_layer.dilation, tuple) else (conv_layer.dilation,)
        stride = conv_layer.stride if isinstance(conv_layer.stride, tuple) else (conv_layer.stride,)

        # from https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        h_out = (h_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
        w_out = (w_in + 2 * padding[-1] - dilation[-1] * (kernel_size[-1] - 1) - 1) // stride[-1] + 1

        return (conv_layer.out_channels, h_out, w_out)
    
    def forward(self, q_matrix):

        #encoding
        encoding = F.relu(self.conv1(q_matrix))
        encoding = F.relu(self.conv2(encoding))
        encoding = self.flatten(encoding)
        message = self.linear1(encoding)

        #decoding
        decoding = F.relu(self.linear2(message))
        decoding = self.unflatten(decoding)
        decoding = F.relu(self.deconv1(decoding))
        q_recon = self.deconv2(decoding)

        return message, q_recon

def trainSAE(autoencoder, training_data, kappa, optim, batch_size, epochs, device):

    dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

    total_losses = []
    recon_losses = []
    sparsity_losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_sparsity_loss = 0.0 

        for q_matrix, _ in dataloader:
            optim.zero_grad(set_to_none = True)
            q_matrix = q_matrix.to(device)

            message, q_recon = autoencoder(q_matrix)

            batch_recon_loss = (1-kappa) * torch.norm(q_matrix - q_recon,2)
            batch_sparsity_loss = kappa * torch.norm(message, 1)
            batch_loss = batch_recon_loss + batch_sparsity_loss

            epoch_loss += batch_loss.detach()
            epoch_recon_loss += batch_recon_loss.detach()
            epoch_sparsity_loss += batch_sparsity_loss.detach()

            batch_loss.backward()
            optim.step()

        total_losses.append(epoch_loss)
        recon_losses.append(epoch_recon_loss)
        sparsity_losses.append(epoch_sparsity_loss)
    autoencoder.to('cpu')
    return torch.tensor(total_losses).cpu(), torch.tensor(recon_losses).cpu(), torch.tensor(sparsity_losses).cpu()





    
        
