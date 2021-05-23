from PIL import Image
import torch
import numpy as np


class KL_DATASET_BASE(torch.utils.data.Dataset):
    def __init__(self, data, bag2indices, bag2prop, transform):
        self.data = data
        self.bag2indices = bag2indices
        self.bag2prop = bag2prop
        self.transform = transform

    def __len__(self):
        return len(self.bag2indices.keys())


class KL_CIFAR10(KL_DATASET_BASE):
    def __getitem__(self, bag_index):
        indices = self.bag2indices[bag_index]
        images = torch.zeros((len(indices), self.data[0].shape[2], self.data[0].shape[0], self.data[0].shape[1]),
                             dtype=torch.float32)
        for i in range(len(indices)):
            idx = indices[i]
            img = self.data[idx]
            img = Image.fromarray(img)
            if self.transform is not None:
                img = self.transform(img)
            images[i] = img
        return images, self.bag2prop[bag_index]


class KL_SVHN(KL_DATASET_BASE):
    def __getitem__(self, bag_index):
        indices = self.bag2indices[bag_index]
        images = torch.zeros((len(indices), self.data[0].shape[0], self.data[0].shape[1], self.data[0].shape[2],),
                             dtype=torch.float32)
        for i in range(len(indices)):
            idx = indices[i]
            img = self.data[idx]
            img = Image.fromarray(np.transpose(img, (1, 2, 0)))
            if self.transform is not None:
                img = self.transform(img)
            images[i] = img
        return images, self.bag2prop[bag_index]


class KL_EMNIST(KL_DATASET_BASE):
    def __init__(self, data, bag2indices, bag2prop, transform):
        super(KL_EMNIST, self).__init__(data, bag2indices, bag2prop, transform)
        img = self.transform(Image.fromarray(self.data[0].numpy(), mode='L'))
        self.new_h = img.shape[1]
        self.new_w = img.shape[2]  # need this for resized emnist

    def __getitem__(self, bag_index):
        indices = self.bag2indices[bag_index]
        images = torch.zeros((len(indices), 1, self.new_h, self.new_w,), dtype=torch.float32)
        for i in range(len(indices)):
            idx = indices[i]
            img = self.data[idx]
            img = Image.fromarray(img.numpy(), mode='L')
            if self.transform is not None:
                img = self.transform(img)
            images[i] = img
        return images, self.bag2prop[bag_index]
