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
train_dataset = N_plus_1_ImageDataset("./mnist", "image_path.txt", "n_plus_1_index_text.txt", train_transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# net = alexnet()
# vgg16 = models.vgg16(num_classes=256)
net = Cnn_32()
model = N_plus_1_net(net).to(device)

# Net.cuda()
# model = Net().to(device)

criterion = my_AngularLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.5)

epochs = 50

# running_loss = 0.0
# Net.train()



def train(epoch):
    model.train()
    for batch_idx, (anchor_imgs, positive_img, negatives_imgs) in enumerate(train_loader):
        negatives_imgs = [negatives_imgs[i] for i in range(list(negatives_imgs.size())[0])]
        # print(negatives_imgs[0].size())
        negatives_imgs = torch.cat(negatives_imgs)
        # print(negatives_imgs.size())
        anchor_imgs, positive_img, negatives_imgs = anchor_imgs.to(device), positive_img.to(device), negatives_imgs.to(device)
        # print(anchor_imgs.size(), negatives_imgs.size())
        optimizer.zero_grad()
        embedded_anchors, embedded_positives, embedded_negatives = model(anchor_imgs, positive_img, negatives_imgs)
        loss = criterion(embedded_anchors, embedded_positives, embedded_negatives)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(anchor_imgs), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def save(epoch):
    filename = "checkpoint.pth.tar"
    directory = "checkpoints/n_plus_1_loss"
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = os.path.join(directory, filename)
    torch.save(model.state_dict(), filename)
for epoch in range(1, epochs+1):
    train(epoch)
    save(epoch)
