from PIL import Image
import os
import os.path

import torch.utils.data
import torchvision.transforms as transforms
import numpy as np


def n_pair_dataset(data_path, transform):
    image_dataset = datasets.ImageFolder(data_path, transform)
    return image_dataset

def default_image_loader(path):
    return Image.open(path)

class N_Pair_ImageDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, filenames_filename, n_pair_file_name, transform,
                 loader=default_image_loader):
        self.base_path = base_path
        self.filenamelist = []
        for line in open(filenames_filename):
            self.filenamelist.append(line.rstrip('\n'))
        paths = []
        for line in open(n_pair_file_name):
            paths.append(([i for i in line.split(",")[0].split()], [i for i in line.split(",")[1].split()])) # ([anchors],[positives])
        self.paths = paths
        self.transform = transform
        self.loader = loader
    def __getitem__(self, index):
        def path2img(path):
            img = self.loader(os.path.join(self.base_path,self.filenamelist[int(path)]))
            # print(img.getextrema())

            return img
        # anchor_imgs = np.array([])
        # positives_imgs = np.array([])
        # for data in self.paths[index]:
        #     print(data)
        #     anchor_imgs.append(self.transform(path2img()))
        #     positives_imgs.append(self.transform(path2img()))

        anchor_imgs = [self.transform(path2img(path)) for path in self.paths[index][0]]
        # print(anchor_imgs)
        positives_imgs = [self.transform(path2img(path)) for path in self.paths[index][1]]
        # print(anchor_imgs)
        anchor_imgs , positives_imgs = torch.stack(anchor_imgs), torch.stack(positives_imgs)
        return anchor_imgs, positives_imgs
    def __len__(self):
        return len(self.paths)


class N_plus_1_ImageDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, filenames_filename, n_plus_1_file_name, transform,
                 loader=default_image_loader):
        self.base_path = base_path
        self.filenamelist = []
        for line in open(filenames_filename):
            self.filenamelist.append(line.rstrip('\n'))
        paths = []
        for line in open(n_plus_1_file_name):
            paths.append((line.split()[0], line.split()[1], line.split()[2:])) # (anchor,positive,[negatives])
        self.paths = paths
        self.transform = transform
        self.loader = loader
    def __getitem__(self, index):

        def path2img(path):
            img = self.loader(os.path.join(self.base_path,self.filenamelist[int(path)]))
            return img

        anchor_img = self.transform(path2img(self.paths[index][0]))# [RGB, 224, 224]
        # anchor_img = torch.unsqueeze(anchor_img, 0)# [1, RGB, 224, 224]
        # print(anchor_img.size())
        positives_img = self.transform(path2img(self.paths[index][1]))# [RGB, 224, 224]
        # positives_img = torch.unsqueeze(positives_img, 0)# [1, RGB, 224, 224]
        negatives_imgs = [self.transform(path2img(path)) for path in self.paths[index][2]]
        negatives_imgs = torch.stack(negatives_imgs)# [N, RGB, 224, 224]
        # print(torch.stack(negatives_imgs).size())
        # negatives_imgs = torch.squeeze(torch.stack(negatives_imgs))

        return anchor_img, positives_img, negatives_imgs
    def __len__(self):
        return len(self.paths)
