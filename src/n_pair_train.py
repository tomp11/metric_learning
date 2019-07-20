import os
import datetime
import glob
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
from models.CNN_3 import CNN_3

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # "metric_test"

traindata_path = os.path.join(base_path, "datasets", "mnist") # "metric_test\datasets\mnist"
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])

# val_dataset = datasets.MNIST(root=testdata_path, train=False, download=True, transform=transforms)

#ImageFolderのdefaultloaderだとmnistなのに3,28,28だったのでpillowのloader使う
def image_loader(path):
    return Image.open(path)
datasets = datasets.ImageFolder(traindata_path, transform, loader=image_loader)

train_size = len(datasets)*0.9
train_dataset = Subset(trainval_dataset, list(range(train_size)))
val_dataset = Subset(trainval_dataset, list(range(train_size, len(datasets))))
batch_sampler = BalancedBatchSampler(train_dataset, n_classes=10, n_samples=8)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=batch_sampler)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_sampler=batch_sampler)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN_3().to(device)
criterion = Angular_mc_loss()
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)


log_base_path = os.path.join(base_path, "logs")
dt = datetime.datetime.now()
model_id = len(glob.glob(os.path.join(log_base_path, "{}{}{}*".format(dt.year, dt.month, dt.day))))
log_dir_name = "{}{}{}_{}_{}".format(dt.year, dt.month, dt.day, '{0:02d}'.format(model_id), model.__class__.__name__)
log_path = os.path.join(log_base_path, log_dir_name)
writer = SummaryWriter(log_dir=log_path)

epochs = 30


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        data, target = data.cuda(), target.cuda()
        embedded = model(data)
        loss = criterion(embedded, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar("loss/train_loss", loss.item(), (len(train_loader)*(epoch-1)+batch_idx))
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                loss.item()))
def val(epoch):
    model.eval()
    with torch.no_grad():

def save(epoch):
    filename = "checkpoint.pth.tar"
    directory = "checkpoints"
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = os.path.join(directory, filename)
    torch.save(model.state_dict(), filename)


if __name__ == "__main__":
    for epoch in range(1, epochs+1):
        train(epoch)
        save(epoch)
