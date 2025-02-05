import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add

from conv import SpiralConv


def Pool(x, trans, dim=1):
    row, col = trans._indices()
    value = trans._values().unsqueeze(-1)
    out = torch.index_select(x, dim, col) * value
    out = scatter_add(out, row, dim, dim_size=trans.size(0))
    return out


class SpiralEnblock(nn.Module):
    def __init__(self, in_channels, out_channels, indices):
        super().__init__()
        self.conv = SpiralConv(in_channels, out_channels, indices)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, down_transform):
        out = F.elu(self.conv(x))
        out = Pool(out, down_transform)
        return out


class SpiralDeblock(nn.Module):
    def __init__(self, in_channels, out_channels, indices):
        super().__init__()
        self.conv = SpiralConv(in_channels, out_channels, indices)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, up_transform):
        out = Pool(x, up_transform)
        out = F.elu(self.conv(out))
        return out

class AEVAE(nn.Module):
    def __init__(self, in_channels, out_channels, latent_channels,
                 spiral_indices, down_transform, up_transform, use_vae=False):
        """
        AEVAE: Autoencoder with optional Variational Autoencoder (VAE) mode.

        :param in_channels: Input feature channels
        :param out_channels: List of output channels for each encoder layer
        :param latent_channels: Size of latent space
        :param spiral_indices: Spiral convolution indices
        :param down_transform: Down-sampling transformations
        :param up_transform: Up-sampling transformations
        :param use_vae: If True, uses VAE; otherwise, standard AE
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.spiral_indices = spiral_indices
        self.down_transform = down_transform
        self.up_transform = up_transform
        self.use_vae = use_vae
        self.num_vert = self.down_transform[-1].size(0)

        # Encoder
        self.en_layers = nn.ModuleList()
        for idx in range(len(out_channels)):
            if idx == 0:
                self.en_layers.append(SpiralEnblock(in_channels, out_channels[idx], self.spiral_indices[idx]))
            else:
                self.en_layers.append(SpiralEnblock(out_channels[idx - 1], out_channels[idx], self.spiral_indices[idx]))

        # VAE: Two separate linear layers for mean and log variance
        if use_vae:
            self.fc_mu = nn.Linear(self.num_vert * out_channels[-1], latent_channels)
            self.fc_logvar = nn.Linear(self.num_vert * out_channels[-1], latent_channels)
        else:
            self.en_layers.append(nn.Linear(self.num_vert * out_channels[-1], latent_channels))

        # Decoder
        self.de_layers = nn.ModuleList()
        self.de_layers.append(nn.Linear(latent_channels, self.num_vert * out_channels[-1]))
        for idx in range(len(out_channels)):
            if idx == 0:
                self.de_layers.append(SpiralDeblock(out_channels[-idx - 1], out_channels[-idx - 1], self.spiral_indices[-idx - 1]))
            else:
                self.de_layers.append(SpiralDeblock(out_channels[-idx], out_channels[-idx - 1], self.spiral_indices[-idx - 1]))

        self.de_layers.append(SpiralConv(out_channels[0], in_channels, self.spiral_indices[0]))
        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_uniform_(param)

    def reparameterize(self, mu, logvar):
        """VAE Reparameterization Trick: z = mu + sigma * epsilon"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encoder(self, x):
        for i, layer in enumerate(self.en_layers):
            if i != len(self.en_layers) - 1:
                x = layer(x, self.down_transform[i])
            else:
                x = x.view(-1, layer.weight.size(1))
                x = layer(x)
        return x

    def decoder(self, x):
        num_layers = len(self.de_layers)
        num_features = num_layers - 2
        for i, layer in enumerate(self.de_layers):
            if i == 0:
                x = layer(x)
                x = x.view(-1, self.num_vert, self.out_channels[-1])
            elif i != num_layers - 1:
                x = layer(x, self.up_transform[num_features - i])
            else:
                x = layer(x)
        return x

    def forward(self, x):
        if self.use_vae:
            encoded = self.encoder(x)
            mu = self.fc_mu(encoded)
            logvar = self.fc_logvar(encoded)
            z = self.reparameterize(mu, logvar)
            reconstructed = self.decoder(z)
            return reconstructed, mu, logvar  # VAE returns extra outputs for KL divergence loss
        else:
            z = self.encoder(x)
            return self.decoder(z)  # Regular AE

    def compute_vae_loss(self, recon_x, x, mu, logvar, beta=1.0):
        """
        Compute the full VAE loss: Reconstruction loss + KL Divergence
        :param recon_x: Reconstructed output
        :param x: Ground truth
        :param mu: Mean of latent distribution
        :param logvar: Log variance of latent distribution
        :param beta: KL weight (default=1.0 for standard VAE)
        """
        recon_loss = F.l1_loss(recon_x, x, reduction='mean')  # Use L1 for sharper reconstructions
        kl_divergence = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())  # Use mean KL
        return recon_loss + beta * kl_divergence, recon_loss, kl_divergence