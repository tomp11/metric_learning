import os
from PIL import Image
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.optim as optim
from torch.autograd import Variable
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter

from modules.Loss import Angular_mc_loss, Angular_mc_loss, N_plus_1_Loss, n_pair_mc_loss
from modules.Sampler import BalancedBatchSampler
from models.cnn32 import Cnn_32

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # "metric_test"

traindata_path = os.path.join(base_path, "datasets", "mnist") # "metric_test\datasets\mnist"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])
# val_dataset =

#ImageFolderのdefaultloaderだとmnistなのに3,28,28だったのでpillowのloader使う
def image_loader(path):
    return Image.open(path)
train_dataset = datasets.ImageFolder(traindata_path, transform, loader=image_loader)
# train_dataset = N_Pair_ImageDataset("./mnist", "image_path.txt", "n_pair_index.txt", train_transform)
# train_batch_sampler = BalancedBatchSampler(train_dataset, n_classes=10, n_samples=64, num_workers=4)
train_batch_sampler = BalancedBatchSampler(train_dataset, n_classes=10, n_samples=8)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Cnn_32().to(device)
criterion = Angular_mc_loss()
optimizer = optim.SGD(model.parameters(), lr=0.0035, momentum=0.9)
# optimizer = optim.SGD(model.parameters(), lr=0.1)


log_path = os.path.join(base_path, "logs", "angular_0.001")
writer = SummaryWriter(log_dir=log_path)

epochs = 3


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # print(data.size())
        optimizer.zero_grad()
        # print(anchor_imgs.size())
        # anchor_imgs, positive_imgs = torch.squeeze(anchor_imgs, dim=0), torch.squeeze(positive_imgs, dim=0)
        # anchor_imgs, positive_imgs = anchor_imgs.cuda(), positive_imgs.cuda()
        data, target = data.cuda(), target.cuda()
        embedded = model(data)
        loss = criterion(embedded, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar("loss", loss.item(), (len(train_loader)*(epoch-1)+batch_idx))
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                loss.item()))
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
