from generator import Generator
from discriminator import Discriminator
from utils import *
import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import argparse

parser = argparse.ArgumentParser(
        description="DCGAN trainer"
    )

parser.add_argument(
    "--epochs",
    type=int,
    help="no. of epochs",
    default=10)

parser.add_argument(
    "--batch_size",
    type=int,
    help="batch size",
    default=16)

parser.add_argument(
    "--device",
    type=str,
    help="GPU/CPU",
    default="cpu")

args = parser.parse_args()

# Creating a Generator instance
gen = Generator()

# Using Binary Cross entropy loss BCE
criterion = nn.BCEWithLogitsLoss()

# Noise dimension
noise_dim = 64

display_step = 500

# batch size
batch_size = args.batch_size

# Learning rate
lr = 0.0002


# Setting as per the paper
beta_1 = 0.5
beta_2 = 0.999

# device cpu/ gpu in which we want ot train GAN
device = args.device

# For data preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

# dataset downloader and loader
try:
    dataloader = DataLoader(
        MNIST('.', download=False, transform=transform),
        batch_size=batch_size,
        shuffle=True)
    print("Dataset found in directory")

except RuntimeError:
    print("Dataset downloading...")
    dataloader = DataLoader(
        MNIST('.', download=True, transform=transform),
        batch_size=batch_size,
        shuffle=True)

gen = Generator(noise_dim).to(device)

gen_opt = torch.optim.Adam(gen.parameters(), lr=lr,
                           betas=(beta_1, beta_2))

disc = Discriminator().to(device)

disc_opt = torch.optim.Adam(disc.parameters(), lr=lr,
                            betas=(beta_1, beta_2))

gen = gen.apply(init_weights)

disc = disc.apply(init_weights)

# Epochs
n_epochs = args.epochs

# Some other loss params
current_step = 0
mean_gen_loss = 0
mean_disc_loss = 0

display_step_bool = False

print("Training started .... ")

for epoch in range(n_epochs):

    print(f"EPOCH:{epoch}/{n_epochs}")

    for real, _ in tqdm(dataloader):

        current_batch_size = len(real)

        real = real.to(device)

        ## updating discriminator ##

        # Initializng zero gradient
        disc_opt.zero_grad()

        # Generating noise vector batch at one pass
        fake_noise = get_noise(current_batch_size,
                               noise_dim,
                               device=device)

        # feeding the noise tensor in generator
        fake = gen(fake_noise)

        disc_fake_pred = disc(fake.detach())

        disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))

        disc_real_pred = disc(real)

        disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))

        disc_loss = (disc_fake_loss + disc_real_loss) / 2

        mean_disc_loss += disc_loss.item()/ display_step

        # update gradients
        disc_loss.backward(retain_graph=True)

        # weight update
        disc_opt.step()

        ## updating generator ##

        # initializing generator gradient to zeros
        gen_opt.zero_grad()

        fake_noise_2 = get_noise(current_batch_size,
                                 noise_dim,
                                 device=device)

        fake_2 = gen(fake_noise_2)

        disc_fake_pred = disc(fake_2)

        gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))

        gen_loss.backward()

        gen_opt.step()

        mean_gen_loss += gen_loss.item() / display_step

        if current_step % display_step == 0 and current_step > 0 and display_step_bool is True:
            print(f"Step {current_step}: Generator loss: {mean_gen_loss}, discriminator loss: {mean_disc_loss}")
            mean_gen_loss = 0
            mean_disc_loss = 0

        else:
            temp_loss = [mean_gen_loss, mean_disc_loss]
            mean_gen_loss = 0
            mean_disc_loss = 0
        current_step += 1

    if display_step_bool is False:
        print(f"Step {current_step}: Generator loss: {temp_loss[0]}, discriminator loss: {temp_loss[1]}")
