import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.optim as optim
from torch.autograd import Variable
import os

from Dataset import N_Pair_ImageDataset, N_plus_1_ImageDataset
from Models import N_PAIR_net, N_plus_1_net
from Loss import Angular_mc_loss, my_AngularLoss, NPairLoss
from AlexNet import alexnet
from git_losses import NPairAngularLoss
import torchvision.models as models
from Nets import Cnn_32

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])
train_dataset = N_Pair_ImageDataset("./mnist", "image_path.txt", "n_pair_index.txt", train_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = Cnn_32()
model = N_PAIR_net(net).to(device)

criterion = Angular_mc_loss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.5)

epochs = 50

def train(epoch):
    model.train()
    for batch_idx, (anchor_imgs, positive_imgs) in enumerate(train_loader):
        optimizer.zero_grad()
        # print(anchor_imgs.size())
        anchor_imgs, positive_imgs = torch.squeeze(anchor_imgs, dim=0), torch.squeeze(positive_imgs, dim=0)
        anchor_imgs, positive_imgs = anchor_imgs.cuda(), positive_imgs.cuda()
        f, f_a = model(anchor_imgs, positive_imgs)
        loss = criterion(f, f_a)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(anchor_imgs), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def save(epoch):
    filename = "checkpoint.pth.tar"
    directory = "checkpoints"
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = os.path.join(directory, filename)
    torch.save(model.state_dict(), filename)

for epoch in range(1, epochs+1):
    train(epoch)
    save(epoch)
