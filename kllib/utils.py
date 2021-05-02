from PIL import Image
import torch


class KL_CIFAR10(torch.utils.data.Dataset):
    def __init__(self, data, bag2indices, bag2prop, transform):
        self.data = data
        self.bag2indices = bag2indices
        self.bag2prop = bag2prop
        self.transform = transform

    def __len__(self):
        return len(self.bag2indices.keys())

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
