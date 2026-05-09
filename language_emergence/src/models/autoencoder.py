# Convolutional Autoencoder, based on cells 13 and 14
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class Autoencoder(nn.Module):


<<<<<<< HEAD
   def __init__(self, in_channels, maze_dim, num_channels, kernel_dim, K):
       super().__init__()


       # encoder layers
       self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=num_channels, kernel_size=kernel_dim, padding=kernel_dim-1)
       self.conv2 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=kernel_dim, padding=kernel_dim-1)
      
       conv1_shape = self.output_dim((in_channels, maze_dim, maze_dim), self.conv1)
       conv2_shape = self.output_dim(conv1_shape, self.conv2)
       flattened_size = conv2_shape[0]*conv2_shape[1]*conv2_shape[2]


       self.flatten = nn.Flatten()
       self.linear1 = nn.Linear(flattened_size, K)
      
       # decoder layers
       self.linear2 = nn.Linear(K, flattened_size)
       self.unflatten = nn.Unflatten(1, conv2_shape)
       self.deconv1 = nn.ConvTranspose2d(in_channels=conv2_shape[0], out_channels=num_channels, kernel_size=kernel_dim, padding=kernel_dim-1)
       self.deconv2 = nn.ConvTranspose2d(in_channels=num_channels, out_channels=in_channels, kernel_size=kernel_dim, padding=kernel_dim-1)
                                        
   def output_dim(self, input_dim, conv_layer):


       h_in = input_dim[-2]
       w_in = input_dim[-1]


       padding = conv_layer.padding if isinstance(conv_layer.padding, tuple) else (conv_layer.padding,)
       kernel_size = conv_layer.kernel_size if isinstance(conv_layer.kernel_size, tuple) else (conv_layer.kernel_size,)
       dilation = conv_layer.dilation if isinstance(conv_layer.dilation, tuple) else (conv_layer.dilation,)
       stride = conv_layer.stride if isinstance(conv_layer.stride, tuple) else (conv_layer.stride,)


       # https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
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


def trainSAE(autoencoder, training_data, gamma_sparse, optim, batch_size, epochs, device, reshape=False):


   dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
   autoencoder.train()


   total_losses = []
   recon_losses = []
   sparsity_losses = []
   all_messages = []


   for _ in range(epochs):


       epoch_loss = 0.0
       epoch_recon_loss = 0.0
       epoch_sparsity_loss = 0.0


       for q_matrices, _ in dataloader:


           q_matrices = q_matrices.to(device)
           if reshape:
               q_matrices = q_matrices.reshape(q_matrices.shape[0], 4, 4, 4)


           optim.zero_grad(set_to_none=True)
           messages, q_recons = autoencoder(q_matrices)


           # batch_recon_loss = (1-kappa) * torch.norm(q_matrices - q_recons, 2)**2
           batch_recon_loss = (1-gamma_sparse) * torch.norm(q_matrices - q_recons, 2)
           batch_sparsity_loss = gamma_sparse * torch.norm(messages, 1)
           batch_loss = batch_recon_loss + batch_sparsity_loss
          
           batch_loss.backward()
           optim.step()
          
           epoch_loss += batch_loss.detach()
           epoch_recon_loss += batch_recon_loss.detach()
           epoch_sparsity_loss += batch_sparsity_loss.detach()
           all_messages.append(messages.detach())


       total_losses.append(epoch_loss)
       recon_losses.append(epoch_recon_loss)
       sparsity_losses.append(epoch_sparsity_loss)


   autoencoder.to('cpu')
   return torch.tensor(total_losses).cpu(), torch.tensor(recon_losses).cpu(), torch.tensor(sparsity_losses).cpu(), all_messages


def testSAE(autoencoder, test_data, batch_size, gamma_sparse, device, reshape=False):


   dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


   total_losses = []
   recon_losses = []
   sparsity_losses = []
   all_messages = []
   message_dict = {}


   autoencoder.to(device)
   autoencoder.eval()


   with torch.no_grad():


       for q_matrices, labels in dataloader:


           q_matrices = q_matrices.to(device)
           if reshape:
               q_matrices = q_matrices.reshape(q_matrices.shape[0], 4, 4, 4)
           messages, q_recons = autoencoder(q_matrices)
           # recon_loss = (1-kappa) * torch.norm(q_matrices - q_recons, 2)**2
           recon_loss = (1-gamma_sparse) * torch.norm(q_matrices - q_recons, 2)
           sparsity_loss = gamma_sparse * torch.norm(messages, 1)
           total_loss = recon_loss + sparsity_loss


           total_losses.append(total_loss)
           recon_losses.append(recon_loss)
           sparsity_losses.append(sparsity_loss)
           all_messages.append(messages.detach())


           # doing this so can more easily pass messages to student in student training
           for i in range(len(labels[0])):
               wall_label = labels[0][i].item()
               init_state = labels[1][i].item()
               goal_state = labels[2][i].item()
               if reshape:
                   demand = labels[3][i].item()
                   message_dict[(wall_label, init_state, goal_state, demand)] = messages[i].detach()
               else:
                   message_dict[(wall_label, init_state, goal_state)] = messages[i].detach()


   return torch.tensor(total_losses), torch.tensor(recon_losses), torch.tensor(sparsity_losses), all_messages, message_dict


def testSAE_inventory(autoencoder, test_data, batch_size, gamma_sparse, device):


   dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


   total_losses = []
   recon_losses = []
   sparsity_losses = []
   all_messages = []
   message_dict = {}


   autoencoder.to(device)
   autoencoder.eval()


   with torch.no_grad():


       for q_matrices, wall_labels, init_states, goal_states, demands in dataloader:


           q_matrices = q_matrices.to(device)
           q_matrices = q_matrices.reshape(q_matrices.shape[0], 4, 4, 4)
           messages, q_recons = autoencoder(q_matrices)
           recon_loss = (1-gamma_sparse) * torch.norm(q_matrices - q_recons, 2)
           sparsity_loss = gamma_sparse * torch.norm(messages, 1)
           total_loss = recon_loss + sparsity_loss


           total_losses.append(total_loss)
           recon_losses.append(recon_loss)
           sparsity_losses.append(sparsity_loss)
           all_messages.append(messages.detach())


           for i in range(len(messages)):
               wall_label = wall_labels[i].item()
               init_state = init_states[i].item()
               goal_state = goal_states[i].item()
               demand = demands[i].item()
               message_dict[(wall_label, init_state, goal_state, demand)] = messages[i].detach()


   return torch.tensor(total_losses), torch.tensor(recon_losses), torch.tensor(sparsity_losses), all_messages, message_dict


def trainSAE_inventory(autoencoder, training_data, gamma_sparse, optim, batch_size, epochs, device):


   dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
   autoencoder.train()


   total_losses = []
   recon_losses = []
   sparsity_losses = []
   all_messages = []


   for _ in range(epochs):


       epoch_loss = 0.0
       epoch_recon_loss = 0.0
       epoch_sparsity_loss = 0.0


       for q_matrices, *_ in dataloader:


           q_matrices = q_matrices.to(device)
           q_matrices = q_matrices.reshape(q_matrices.shape[0], 4, 4, 4)


           optim.zero_grad(set_to_none=True)
           messages, q_recons = autoencoder(q_matrices)


           batch_recon_loss = (1-gamma_sparse) * torch.norm(q_matrices - q_recons, 2)
           batch_sparsity_loss = gamma_sparse * torch.norm(messages, 1)
           batch_loss = batch_recon_loss + batch_sparsity_loss


           batch_loss.backward()
           optim.step()


           epoch_loss += batch_loss.detach()
           epoch_recon_loss += batch_recon_loss.detach()
           epoch_sparsity_loss += batch_sparsity_loss.detach()
           all_messages.append(messages.detach())


       total_losses.append(epoch_loss)
       recon_losses.append(epoch_recon_loss)
       sparsity_losses.append(epoch_sparsity_loss)


   autoencoder.to('cpu')
   return torch.tensor(total_losses).cpu(), torch.tensor(recon_losses).cpu(), torch.tensor(sparsity_losses).cpu(), all_messages    

=======
        # encoder layers
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=num_channels, kernel_size=kernel_dim, padding=kernel_dim-1)
        self.conv2 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=kernel_dim, padding=kernel_dim-1)
        
        conv1_shape = self.output_dim((in_channels, maze_dim, maze_dim), self.conv1)
        conv2_shape = self.output_dim(conv1_shape, self.conv2)
        flattened_size = conv2_shape[0]*conv2_shape[1]*conv2_shape[2]

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(flattened_size, K)
        
        # decoder layers
        self.linear2 = nn.Linear(K, flattened_size)
        self.unflatten = nn.Unflatten(1, conv2_shape)
        self.deconv1 = nn.ConvTranspose2d(in_channels=conv2_shape[0], out_channels=num_channels, kernel_size=kernel_dim, padding=kernel_dim-1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=num_channels, out_channels=in_channels, kernel_size=kernel_dim, padding=kernel_dim-1)
                                          
    def output_dim(self, input_dim, conv_layer):

        h_in = input_dim[-2]
        w_in = input_dim[-1]

        padding = conv_layer.padding if isinstance(conv_layer.padding, tuple) else (conv_layer.padding,)
        kernel_size = conv_layer.kernel_size if isinstance(conv_layer.kernel_size, tuple) else (conv_layer.kernel_size,)
        dilation = conv_layer.dilation if isinstance(conv_layer.dilation, tuple) else (conv_layer.dilation,)
        stride = conv_layer.stride if isinstance(conv_layer.stride, tuple) else (conv_layer.stride,)

        # https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
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

def trainSAE(autoencoder, training_data, gamma_sparse, optim, batch_size, epochs, device, reshape=False):

    dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    autoencoder.train()

    total_losses = []
    recon_losses = []
    sparsity_losses = []
    all_messages = []

    for _ in range(epochs):

        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_sparsity_loss = 0.0 

        for q_matrices, _ in dataloader:

            q_matrices = q_matrices.to(device)
            if reshape:
                q_matrices = q_matrices.reshape(q_matrices.shape[0], 4, 4, 4)

            optim.zero_grad(set_to_none=True)
            messages, q_recons = autoencoder(q_matrices)

            # batch_recon_loss = (1-kappa) * torch.norm(q_matrices - q_recons, 2)**2
            batch_recon_loss = (1-gamma_sparse) * torch.norm(q_matrices - q_recons, 2)
            batch_sparsity_loss = gamma_sparse * torch.norm(messages, 1)
            batch_loss = batch_recon_loss + batch_sparsity_loss
            
            batch_loss.backward()
            optim.step()
            
            epoch_loss += batch_loss.detach()
            epoch_recon_loss += batch_recon_loss.detach()
            epoch_sparsity_loss += batch_sparsity_loss.detach()
            all_messages.append(messages.detach())

        total_losses.append(epoch_loss)
        recon_losses.append(epoch_recon_loss)
        sparsity_losses.append(epoch_sparsity_loss)

    autoencoder.to('cpu')
    return torch.tensor(total_losses).cpu(), torch.tensor(recon_losses).cpu(), torch.tensor(sparsity_losses).cpu(), all_messages

def testSAE(autoencoder, test_data, batch_size, gamma_sparse, device, reshape=False):

    dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    total_losses = []
    recon_losses = []
    sparsity_losses = []
    all_messages = []
    message_dict = {}

    autoencoder.to(device)
    autoencoder.eval()

    with torch.no_grad():

        for q_matrices, labels in dataloader:

            q_matrices = q_matrices.to(device)
            if reshape:
                q_matrices = q_matrices.reshape(q_matrices.shape[0], 4, 4, 4)
            messages, q_recons = autoencoder(q_matrices)
            # recon_loss = (1-kappa) * torch.norm(q_matrices - q_recons, 2)**2
            recon_loss = (1-gamma_sparse) * torch.norm(q_matrices - q_recons, 2)
            sparsity_loss = gamma_sparse * torch.norm(messages, 1)
            total_loss = recon_loss + sparsity_loss

            total_losses.append(total_loss)
            recon_losses.append(recon_loss)
            sparsity_losses.append(sparsity_loss)
            all_messages.append(messages.detach())

            # doing this so can more easily pass messages to student in student training
            for i in range(len(messages)):
                wall_label = labels[0][i].item()
                init_state = labels[1][i].item()
                goal_state = labels[2][i].item()
                if reshape:
                    demand = labels[3][i].item()
                    message_dict[(wall_label, init_state, goal_state, demand)] = messages[i].detach()
                else:
                    message_dict[(wall_label, init_state, goal_state)] = messages[i].detach()

    return torch.tensor(total_losses), torch.tensor(recon_losses), torch.tensor(sparsity_losses), all_messages, message_dict
>>>>>>> ea86e4b56e24de0a40c62697ea61c9c6b27d24af


