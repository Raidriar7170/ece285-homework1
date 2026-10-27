"""Convolutional variational autoencoder for 64 x 64 grayscale images."""
import torch
from torch import nn
from torch.nn import functional as F


class VAE(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        channels = [1, 32, 64, 128, 256]
        layers = []
        for a, b in zip(channels[:-1], channels[1:]):
            layers += [nn.Conv2d(a, b, 4, 2, 1), nn.ReLU()]
        self.encoder = nn.Sequential(*layers)
        self.mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.logvar = nn.Linear(256 * 4 * 4, latent_dim)
        self.project = nn.Linear(latent_dim, 256 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
        )

    def encode(self, x):
        h = self.encoder(x).flatten(1)
        return self.mu(h), self.logvar(h)

    def decode(self, z):
        return self.decoder(self.project(z).reshape(-1, 256, 4, 4))

    def forward(self, x, sample=True):
        mu, logvar = self.encode(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu) if sample else mu
        return self.decode(z), mu, logvar


def loss_terms(logits, target, mu, logvar):
    """Per-image pixel-summed BCE + latent-summed KL, averaged over batch."""
    reconstruction = F.binary_cross_entropy_with_logits(logits, target, reduction='none').flatten(1).sum(1).mean()
    kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(1).mean()
    return reconstruction + kl, reconstruction, kl
