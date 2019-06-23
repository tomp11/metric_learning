from __future__ import print_function
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.decomposition import PCA

from Nets import Cnn_32
from Models import Test_net


def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_set = datasets.MNIST(root='./mnist_test', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]))

    test_batch = []
    target = []
    count = 0
    for i, j in test_set:
        test_batch.append(i)
        target.append(j)
        count+=1
        # print(i.size())
        if count==1000:
            break
    data = torch.stack(test_batch) # (1000,1,28,28)
    target = torch.Tensor(target) # (1000)
    # print(target)


    # print(batch.size())

    # test_loader = torch.utils.data.DataLoader(
    # datasets.MNIST(root='./mnist_test', train=False, download=True, transform=transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.1307,), (0.3081,))
    # ])), batch_size=1000, shuffle=True, num_workers=4)

    net = Cnn_32()
    model = Test_net(net).to(device)
    model.load_state_dict(torch.load('./checkpoints/checkpoint.pth.tar'))
    classes = ["0","1","2","3","4","5","6","7","8","9"]


    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        pred_categories = [] #予想ラベルたち

        target_array = target.numpy()
        master_features = [] #マスター画像の埋め込みたち
        for i in classes:
            indexes = np.where(target_array==int(i))[0]
            master_img = data[np.random.choice(indexes)].to(device)
            master_img = torch.unsqueeze(master_img, dim=0)
            embedded_master_img = model(master_img)
            master_features.append(embedded_master_img)
        master_features = torch.cat(master_features) # (10, 128)
        print(master_features.size())

    # data, target = data.to(device), target.to(device)
    data = data.to(device)
    output = model(data)
    output = torch.unbind(output)
    for embedded_img in output:
        distances = torch.sum((master_features - embedded_img)**2, dim=1) #(10)
        pred_category = classes[distances.argmin()]
        pred_categories.append(int(pred_category))#
    pred_category = torch.Tensor(pred_categories)

    # ラベルが数字だったのでtorch.Tensorにして条件文でやった。strならforぶん回す

    correct += (target == pred_category).sum()
    accuracy = float(correct)*100 / len(pred_categories)


    print('Accuracy: {}/{} ({}%)\n'.format(correct, len(pred_categories), accuracy))


def auto_encode(model, device, data):
    model.eval()
    with torch.no_grad():
        data = data.to(device)
        latent_vecs = model(data)
    return latent_vecs

def t_sne():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='./mnist_test', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])), batch_size=1000, shuffle=True, num_workers=4)
    test_set = datasets.MNIST(root='./mnist_test', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]))

    test_batch = []
    target = []
    count = 0
    for i, j in test_set:
        test_batch.append(i)
        target.append(j)
        count+=1
        # print(i.size())
        if count==1000:
            break
    data = torch.stack(test_batch) # (1000,1,28,28)
    target = torch.Tensor(target) # (1000)

    net = Cnn_32()
    model = Test_net(net).to(device)


    # Auto Encode using trained model
    model.load_state_dict(torch.load('./checkpoints/checkpoint.pth.tar'))
    latent_vecs = auto_encode(model, device, data)
    latent_vecs, target = latent_vecs.to("cpu"), target.to("cpu")
    latent_vecs, target = latent_vecs.numpy(), target.numpy()
    print(latent_vecs.shape, target.shape)
    latent_vecs_reduced = TSNE(n_components=2, random_state=0).fit_transform(latent_vecs[:1000])
    # latent_vecs_reduced = PCA(n_components=2).fit_transform(latent_vecs)

    plt.scatter(latent_vecs_reduced[:, 0], latent_vecs_reduced[:, 1],
                c=target[:1000], cmap='jet')
    plt.colorbar()
    plt.show()

if __name__ == '__main__':
    t_sne()
    # test()
