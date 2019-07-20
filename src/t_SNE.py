from __future__ import print_function
import os
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

from models.CNN_3 import CNN_3



def test(t_SNE=True):
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    testdata_path = os.path.join(base_path, "datasets", "mnist_test")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_set = datasets.MNIST(root=testdata_path, train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]))

    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=5000, shuffle=False)
    # 1バッチ分だけ取り出し
    # testセット10000の内後半の5000取り出し
    data_iter = iter(test_loader)
    _, _ = data_iter.next()
    data, target = data_iter.next() # (5000,1,28,28), (5000)
    target = torch.LongTensor(target)
    print(data.size(), target.size())
    # data = torch.stack([test_set[i][0] for i in range(5000)]) # (5000,1,28,28)
    # target = torch.Tensor([test_set[i][1] for i in range(5000)]) # (5000)

    net = CNN_3()
    model = net.to(device)
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
            master_img = master_img.to(device)
            embedded_master_img = model(master_img)
            master_features.append(embedded_master_img)
        master_features = torch.cat(master_features) # (10, 128)

        data = data.to(device)
        output = model(data)
        output_unbind = torch.unbind(output)
    for embedded_img in output_unbind:
        distances = torch.sum((master_features - embedded_img)**2, dim=1) #(10)
        pred_category = classes[distances.argmin()]
        pred_categories.append(int(pred_category))#
    pred_category = torch.LongTensor(pred_categories)
    # ラベルが数字だったのでtorch.Tensorにして条件文でやった。strならforぶん回す
    correct += (target == pred_category).sum()
    accuracy = float(correct)*100 / len(pred_categories)

    print('Accuracy: {}/{} ({}%)\n'.format(correct, len(pred_categories), accuracy))

    if t_SNE:
        t_sne(output, target)


def t_sne(latent_vecs, target):
    latent_vecs = latent_vecs.to("cpu")
    latent_vecs = latent_vecs.numpy()
    latent_vecs_reduced = TSNE(n_components=2, random_state=0).fit_transform(latent_vecs)
    # latent_vecs_reduced = PCA(n_components=2).fit_transform(latent_vecs)
    plt.scatter(latent_vecs_reduced[:, 0], latent_vecs_reduced[:, 1],
                c=target, cmap='jet')
    plt.colorbar()
    plt.show()

if __name__ == '__main__':
    test()
