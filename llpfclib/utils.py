import torch
from PIL import Image


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
		img = Image.fromarray(img, mode='L')
		if self.transform is not None:
			img = self.transform(img)
		return img, int(y_), torch.tensor(gamma_m, dtype=None)


class FORWARD_CORRECT_CIFAR10(torch.utils.data.Dataset):
	def __init__(self, data, noisy_y, group2transition, group2weights, instance2group, transform):
		self.data = data
		self.noisy_y = noisy_y
		self.group2transition = group2transition
		self.group2weights = group2weights
		self.instance2group = instance2group
		self.transform = transform

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		img, y_ = self.data[index], self.noisy_y[index]
		trans_m = self.group2transition[self.instance2group[index]]
		weight = self.group2weights[self.instance2group[index]]
		img = Image.fromarray(img)
		if self.transform is not None:
			img = self.transform(img)
		return img, int(y_), torch.tensor(trans_m, dtype=None), weight
