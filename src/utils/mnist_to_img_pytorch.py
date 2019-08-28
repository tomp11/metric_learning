import os
from PIL import Image
import torch
from torchvision import datasets
import numpy as np

base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
dataset_path = os.path.join(base_path, "datasets")

mnist_train_path = os.path.join(dataset_path, "mnist_train")
metric_mnist_train_path = os.path.join(dataset_path, "metric_mnist_train")
metric_mnist_val_path = os.path.join(dataset_path, "metric_mnist_val")

mnist_train =  datasets.MNIST(root=mnist_train_path, train=True, download=True)

if not os.path.isdir(dataset_path):
    os.mkdir(dataset_path)
if not os.path.isdir(mnist_train_path):
    os.mkdir(mnist_train_path)
if not os.path.isdir(metric_mnist_train_path):
    os.mkdir(metric_mnist_train_path)
if not os.path.isdir(metric_mnist_val_path):
    os.mkdir(metric_mnist_val_path)
for i in range(10):
    dirname = str(i)
    if not os.path.isdir(os.path.join(metric_mnist_train_path, dirname)):
        os.mkdir(os.path.join(metric_mnist_train_path, dirname))
    if not os.path.isdir(os.path.join(metric_mnist_val_path, dirname)):
        os.mkdir(os.path.join(metric_mnist_val_path, dirname))

def save(data, target, savepath, index):
    filename = os.path.join(savepath, str(target), "train{0:04d}.png".format(index))
    data.save(filename)
    print(os.path.join(str(target), "train{0:04d}.png".format(index)))

for i in range(len(mnist_train)*9//10):
    data, target = mnist_train[i]
    save(data, target, metric_mnist_train_path, i)
for i in range(len(mnist_train)*9//10, len(mnist_train)):
    data, target = mnist_train[i]
    save(data, target, metric_mnist_val_path, i)
