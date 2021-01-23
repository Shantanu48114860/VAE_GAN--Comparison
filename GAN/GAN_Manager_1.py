import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from tqdm import tqdm

from GAN import Generator, Discriminator


class Gan_Manager:
    def train_GAN(self, data_loader_train, device):
        lr = 0.0002
        netG = Generator().to(device)
        netD = Discriminator().to(device)

        # Initialize BCELoss function
        criterion = nn.BCELoss()

        # Create batch of latent vectors that we will use to visualize
        #  the progression of the generator
        nz = 100
        fixed_noise = torch.randn(64, nz, 1, 1, device=device)
        beta1 = 0.5

        # Establish convention for real and fake labels during training
        real_label = 1.
        fake_label = 0.

        # Setup Adam optimizers for both G and D
        optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
        optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

        # Training Loop

        # Lists to keep track of progress
        img_list = []
        G_losses = []
        D_losses = []
        iters = 0
        num_epochs = 150

        print("Starting Training Loop...")
        # For each epoch
        for epoch in range(num_epochs):
            # For each batch in the dataloader
            with tqdm(total=len(train_data_loader)) as t:
                for i, data in enumerate(data_loader_train, 0):

                    ############################
                    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                    ###########################
                    ## Train with all-real batch
                    netD.zero_grad()
                    # Format batch
                    real_cpu = data[0].to(device)
                    b_size = real_cpu.size(0)
                    label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
                    # Forward pass real batch through D
                    output_real = netD(real_cpu).view(-1)

                    # Generate batch of latent vectors
                    noise = torch.randn(b_size, nz, 1, 1, device=device)
                    # Generate fake image batch with G
                    fake = netG(noise)
                    label.fill_(fake_label)
                    # Classify all fake batch with D
                    output_fake = netD(fake).view(-1)
                    # Calculate D's loss on the all-fake batch
                    errD_fake = - torch.mean(torch.log(output_real) + torch.log(1 - output_fake))
                    # Calculate the gradients for this batch
                    errD_fake.backward(retain_graph=True)
                    # Add the gradients from the all-real and all-fake batches
                    errD = errD_fake
                    # Update D
                    optimizerD.step()

                    ############################
                    # (2) Update G network: maximize log(D(G(z)))
                    ###########################
                    # Calculate G's loss based on this output
                    output_fake = netD(fake).view(-1)
                    errG = - torch.mean(torch.log(output_fake))
                    # Calculate gradients for G
                    errG.backward()
                    # Update G
                    optimizerG.step()

                    # Output training stats
                    t.set_postfix(epoch='{0}'.format(epoch),
                                  loss_g='{:05.3f}'.format(errG.item()),
                                  loss_d='{:05.3f}'.format(errD.item()))
                    t.update()
                    # if i % 50 == 0:
                    #     print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    #           % (epoch, num_epochs, i, len(data_loader_train),
                    #              errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

                    # Save Losses for plotting later
                    G_losses.append(errG.item())
                    D_losses.append(errD.item())

                    # Check how the generator is doing by saving G's output on fixed_noise
                    if (iters % 10 == 0) or ((epoch == num_epochs - 1) and (i == len(data_loader_train) - 1)):
                        with torch.no_grad():
                            fake = netG(fixed_noise).detach().cpu()
                        img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

                    iters += 1

        return G_losses, D_losses, img_list

    @staticmethod
    def plot_loss(G_losses, D_losses, fig_name):
        plt.figure(figsize=(10, 5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(G_losses, label="G")
        plt.plot(D_losses, label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        # plt.show()
        plt.draw()
        plt.savefig(fig_name, dpi=220)
        plt.clf()

    @staticmethod
    def plot_image(test_data_loader, img_list, fig_name):
        # Grab a batch of real images from the dataloader
        real_batch = next(iter(test_data_loader))

        # Plot the real images
        plt.figure(figsize=(15, 15))
        plt.subplot(1, 2, 1)
        plt.axis("off")
        plt.title("Real Images")
        plt.imshow(
            np.transpose(vutils.make_grid(real_batch[0].to(device)[:64],
                                          padding=5, normalize=True).cpu(), (1, 2, 0)))

        # Plot the fake images from the last epoch
        plt.subplot(1, 2, 2)
        plt.axis("off")
        plt.title("Fake Images")
        plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
        plt.draw()
        plt.savefig(fig_name, dpi=220)
        plt.clf()


if __name__ == "__main__":
    print("---GAN---")
    img_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    batch_size = 128
    train_dataset = MNIST(root='./data/MNIST', download=True, train=True,
                          transform=img_transform)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = MNIST(root='./data/MNIST', download=True, train=False, transform=img_transform)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    gan = Gan_Manager()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("### Device: {0} ###".format(device))

    G_losses, D_losses, img_list = gan.train_GAN(train_data_loader, device)

    gan.plot_loss(G_losses, D_losses, fig_name="./Plots/Loss_plot.jpeg")
    gan.plot_image(test_data_loader, img_list, fig_name="./Plots/Real vs Fake.jpeg")
