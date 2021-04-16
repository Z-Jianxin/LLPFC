import torch
from PIL import Image
import numpy as np


def truncate_data(data, bag2indices):
	idx_list = []
	for bag_id in bag2indices.keys():
		idx_list.extend(bag2indices[bag_id])
	idx_list.sort()
	data_truncated = data[idx_list]
	idx2new = {idx_list[i]: i for i in range(len(idx_list))}
	bag2new = {bag_id: list(map(idx2new.get, bag2indices[bag_id])) for bag_id in bag2indices.keys()}
	return data_truncated, bag2new


class FORWARD_CORRECT_MNIST(torch.utils.data.Dataset):
	def __init__(self, data, noisy_y, group2gamma, instance2group, transform):
		self.data = data
		self.noisy_y = noisy_y
		self.group2gamma = group2gamma
		self.instance2group = instance2group
		self.transform = transform

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		img, y_, gamma_m = self.data[index], self.noisy_y[index], self.group2gamma[self.instance2group[index]]
		#img = Image.fromarray(img.numpy(), mode='L')
		img = Image.fromarray(img, mode='L')
		if self.transform is not None:
			img = self.transform(img)
		return img, int(y_), torch.tensor(gamma_m, dtype=None)


class FORWARD_CORRECT_CIFAR10(torch.utils.data.Dataset):
	def __init__(self, data, noisy_y, group2gamma, instance2group, transform):
		self.data = data
		self.noisy_y = noisy_y
		self.group2gamma = group2gamma
		self.instance2group = instance2group
		self.transform = transform

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		img, y_, gamma_m = self.data[index], self.noisy_y[index], self.group2gamma[self.instance2group[index]]
		img = Image.fromarray(img)
		if self.transform is not None:
			img = self.transform(img)
		return img, int(y_), torch.tensor(gamma_m, dtype=None)
