import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        nz = 100
        ngf = 64
        nc = 1

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(in_channels=nz, out_channels=ngf * 8,
                               kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 3 x 3

            nn.ConvTranspose2d(in_channels=ngf * 8,
                               out_channels=ngf * 4,
                               kernel_size=3, stride=2,
                               padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 5 x 5

            nn.ConvTranspose2d(in_channels=ngf * 4,
                               out_channels=ngf * 2,
                               kernel_size=5, stride=1,
                               padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 7 x 7

            nn.ConvTranspose2d(in_channels=ngf * 2,
                               out_channels=ngf,
                               kernel_size=4, stride=2,
                               padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 14 x 14

            nn.ConvTranspose2d(in_channels=ngf,
                               out_channels=nc,
                               kernel_size=4, stride=2,
                               padding=1, bias=False),
            nn.Tanh()
            # state size. (nc) x 28 x 28
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        self.fc1 = nn.Linear(in_features=12 * 4 * 4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.fc3 = nn.Linear(in_features=60, out_features=30)
        self.out = nn.Linear(in_features=30, out_features=1)

    def forward(self, t):
        # input layer
        t = t

        # 1st conv layer
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # 2nd conv layer
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # 3rd FC1
        t = self.fc1(t.reshape(-1, 12 * 4 * 4))
        t = F.relu(t)

        # 4th FC2
        t = self.fc2(t)
        t = F.relu(t)

        t = self.fc3(t)
        t = F.relu(t)

        # output layer
        t = self.out(t)
        t = torch.sigmoid(t)

        return t


netG = Generator()
netD = Discriminator()

nz = 100
print(netG)
print(netD)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# batch_size = 128
# img_transform = transforms.Compose([
#     transforms.ToTensor()
# ])
# train_dataset = MNIST(root='./data/MNIST', download=True, train=True,
#                       transform=img_transform)
# train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# data, labels = iter(train_data_loader).next()
#
# real_cpu = data[0].to(device)
# b_size = real_cpu.size(0)
# noise = torch.randn(b_size, nz, 1, 1, device=device)
#
# print(noise.size())
# fake = netG(noise)
# print(fake.size())
#
# prob = netD(data)
# print(prob.size())
# # print(prob)
