import torch
from torch import nn


class Generator(nn.Module):

    def __init__(self,
                 noise_dim=10,
                 img_channels=1,
                 hidden_dim=64):

        super(Generator, self).__init__()
        self.noise_dim = noise_dim

        self.gen = nn.Sequential(
            nn.ConvTranspose2d(noise_dim, hidden_dim*4,
                               kernel_size=3, stride=2),
            nn.BatchNorm2d(hidden_dim*4),
            nn.ReLU(),

            nn.ConvTranspose2d(hidden_dim*4, hidden_dim*2,
                               kernel_size=4, stride=1),
            nn.BatchNorm2d(hidden_dim*2),
            nn.ReLU(),

            nn.ConvTranspose2d(hidden_dim*2, hidden_dim,
                               kernel_size=3, stride=2),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),

            nn.ConvTranspose2d(hidden_dim, img_channels, kernel_size=4, stride=2),
            nn.Tanh()
        )

    def forward(self, noise):
        x = noise.view(len(noise), self.noise_dim, 1, 1)
        return self.gen(x)
