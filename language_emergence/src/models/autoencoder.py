# src/models/autoencoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class Autoencoder(nn.Module):

    def __init__(self, in_channels, maze_dim, num_channels, kernel_dim, K):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, num_channels, kernel_dim, padding=kernel_dim - 1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_dim, padding=kernel_dim - 1)

        s1 = self._out_shape((in_channels, maze_dim, maze_dim), self.conv1)
        s2 = self._out_shape(s1, self.conv2)
        flat = s2[0] * s2[1] * s2[2]

        self.flatten  = nn.Flatten()
        self.linear1  = nn.Linear(flat, K)

        self.linear2  = nn.Linear(K, flat)
        self.unflatten = nn.Unflatten(1, s2)
        self.deconv1  = nn.ConvTranspose2d(s2[0], num_channels, kernel_dim, padding=kernel_dim - 1)
        self.deconv2  = nn.ConvTranspose2d(num_channels, in_channels, kernel_dim, padding=kernel_dim - 1)

    @staticmethod
    def _out_shape(input_shape, layer):
        c_in, h, w = input_shape
        pad    = layer.padding  if isinstance(layer.padding,   tuple) else (layer.padding,)
        ks     = layer.kernel_size if isinstance(layer.kernel_size, tuple) else (layer.kernel_size,)
        dil    = layer.dilation if isinstance(layer.dilation,  tuple) else (layer.dilation,)
        stride = layer.stride   if isinstance(layer.stride,    tuple) else (layer.stride,)
        h_out  = (h + 2*pad[0]  - dil[0]  * (ks[0]  - 1) - 1) // stride[0]  + 1
        w_out  = (w + 2*pad[-1] - dil[-1] * (ks[-1] - 1) - 1) // stride[-1] + 1
        return (layer.out_channels, h_out, w_out)

    def forward(self, q_matrix):
        x = F.relu(self.conv1(q_matrix))
        x = F.relu(self.conv2(x))
        x = self.flatten(x)
        message = self.linear1(x)

        x = F.relu(self.linear2(message))
        x = self.unflatten(x)
        x = F.relu(self.deconv1(x))
        q_recon = self.deconv2(x)
        return message, q_recon


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def trainSAE(autoencoder, training_data, gamma_sparse, optim, batch_size, epochs, device):
    """
    Train the sparse autoencoder.

    Dataset must yield batches of:
        (q_matrices, label_field_0, label_field_1, ...)
    where q_matrices are already shaped (batch, n_actions, grid_dim, grid_dim).
    """
    dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    autoencoder.train()

    total_losses, recon_losses, sparsity_losses, all_messages = [], [], [], []

    for _ in range(epochs):
        ep_loss = ep_recon = ep_sparse = 0.0

        for batch in dataloader:
            q_matrices = batch[0].to(device)            # first element always q-matrices
            
            optim.zero_grad(set_to_none=True)
            messages, q_recons = autoencoder(q_matrices)

            recon_loss   = (1 - gamma_sparse) * torch.norm(q_matrices - q_recons, 2)
            sparse_loss  = gamma_sparse       * torch.norm(messages, 1)
            loss         = recon_loss + sparse_loss

            loss.backward()
            optim.step()

            ep_loss   += loss.detach()
            ep_recon  += recon_loss.detach()
            ep_sparse += sparse_loss.detach()
            all_messages.append(messages.detach())

        total_losses.append(ep_loss)
        recon_losses.append(ep_recon)
        sparsity_losses.append(ep_sparse)

    autoencoder.to('cpu')
    return (
        torch.tensor(total_losses).cpu(),
        torch.tensor(recon_losses).cpu(),
        torch.tensor(sparsity_losses).cpu(),
        all_messages,
    )


# ---------------------------------------------------------------------------
# Testing / message extraction
# ---------------------------------------------------------------------------

def testSAE(autoencoder, test_data, batch_size, gamma_sparse, device):
    dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    autoencoder.to(device)
    autoencoder.eval()

    total_losses, recon_losses, sparsity_losses, all_messages = [], [], [], []
    message_dict = {}

    with torch.no_grad():
        for q_matrices, labels in dataloader:
            q_matrices = q_matrices.to(device)

            # Forward pass
            messages, q_recons = autoencoder(q_matrices)

            # Loss calculation
            recon_loss  = (1 - gamma_sparse) * torch.norm(q_matrices - q_recons, 2)
            sparse_loss = gamma_sparse * torch.norm(messages, 1)
            total_loss  = recon_loss + sparse_loss

            total_losses.append(total_loss)
            recon_losses.append(recon_loss)
            sparsity_losses.append(sparse_loss)
            all_messages.append(messages.detach())

            for i in range(len(messages)):
                # Extract mandatory 3-tuple components
                wall_idx = int(labels[0][i].item())
                init_st  = int(labels[1][i].item())
                goal_st  = int(labels[2][i].item())
                
                # Check if a fourth label (order_mode_id) exists
                if len(labels) > 3:
                    order_mode = int(labels[3][i].item())
                    # Construct a 4-tuple key
                    key = (wall_idx, init_st, goal_st, order_mode)
                else:
                    # Fallback to 3-tuple if no order mode is provided
                    key = (wall_idx, init_st, goal_st)
                
                message_dict[key] = messages[i].detach()

    return (
        torch.tensor(total_losses),
        torch.tensor(recon_losses),
        torch.tensor(sparsity_losses),
        all_messages,
        message_dict,
    )