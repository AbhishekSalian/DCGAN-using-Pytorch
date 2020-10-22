from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torch
from torch import nn


def show_images(img_tensor, num_images=25, size=(1, 28, 28)):
    """This function helps in visualizing image grid given a tensor

    Args:
        img_tensor (tensor): [description]
        num_images (int, optional): [description]. Defaults to 25.
        size (tuple, optional): [description]. Defaults to (1, 28, 28).
    """
    img_tensor = (img_tensor + 1) / 2
    img_unflat = img_tensor.detach().cpu()
    img_grid = make_grid(img_unflat[:num_images], nrow=5)
    plt.imshow(img_grid.premute(1, 2, 0).squeeze())
    plt.show()


def get_noise(n_samples, noise_dim, device="cpu"):
    return torch.randn(n_samples, noise_dim, device=device)


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
 
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)
