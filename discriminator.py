import torch
from torch import nn


class Discriminator(nn.Module):
    def __init__(self, img_channels=1, hidden_dim=16):
        super(Discriminator, self).__init__()

        self.disc = nn.Sequential(

            nn.Conv2d(in_channels=img_channels,
                      out_channels=hidden_dim,
                      kernel_size=4,
                      stride=2),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(in_channels=hidden_dim,
                      out_channels=hidden_dim*2,
                      kernel_size=4,
                      stride=2),
            nn.BatchNorm2d(hidden_dim*2),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(in_channels=hidden_dim*2,
                      out_channels=1,
                      kernel_size=4,
                      stride=2) 

        )

    def forward(self, image):
        disc_pred = self.disc(image)
        return disc_pred.view(len(disc_pred), -1)
