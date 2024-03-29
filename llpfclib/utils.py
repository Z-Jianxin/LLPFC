import torch
from PIL import Image
from torch.utils.data import Sampler
import numpy as np


def truncate_data_group(x, y, instance2group):
	idx_list = []
	for i in range(x.shape[0]):
		if instance2group[i] != -1:
			idx_list.append(i)
	x_truncated = x[idx_list]
	y_truncated = y[idx_list]
	idx2new = {idx_list[i]: i for i in range(len(idx_list))}
	instance2group_new = {}
	for old, new in idx2new.items():
		instance2group_new[new] = instance2group[old]
	new2idx = {idx2new[idx]: idx for idx in idx2new.keys()}
	return x_truncated, y_truncated, instance2group_new, new2idx


class LLPFC_DATASET_BASE(torch.utils.data.Dataset):
	def __init__(self, data, noisy_y, group2transition, instance2weight, instance2group, transform):
		self.data, self.noisy_y, self.instance2group, self.new2idx = truncate_data_group(data, noisy_y, instance2group)
		self.group2transition = group2transition
		self.instance2weight = instance2weight
		self.transform = transform

	def __len__(self):
		return len(self.data)


class FORWARD_CORRECT_CIFAR10(LLPFC_DATASET_BASE):
	def __getitem__(self, index):
		img, y_ = self.data[index], self.noisy_y[index]
		trans_m = self.group2transition[self.instance2group[index]]
		weight = self.instance2weight[self.new2idx[index]]
		img = Image.fromarray(img)
		if self.transform is not None:
			img = self.transform(img)
		return img, int(y_), torch.tensor(trans_m, dtype=None), weight


class FORWARD_CORRECT_SVHN(LLPFC_DATASET_BASE):
	def __getitem__(self, index):
		img, y_ = self.data[index], self.noisy_y[index]
		img = Image.fromarray(np.transpose(img, (1, 2, 0)))
		trans_m = self.group2transition[self.instance2group[index]]
		weight = self.instance2weight[self.new2idx[index]]
		if self.transform is not None:
			img = self.transform(img)
		return img, int(y_), torch.tensor(trans_m, dtype=None), weight


class FORWARD_CORRECT_MNIST(LLPFC_DATASET_BASE): # this should work for both EMNIST and MNIST
	def __getitem__(self, index):
		img, y_ = self.data[index], self.noisy_y[index]
		trans_m = self.group2transition[self.instance2group[index]]
		weight = self.instance2weight[self.new2idx[index]]
		img = Image.fromarray(img.numpy(), mode='L')
		if self.transform is not None:
			img = self.transform(img)
		return img, int(y_), torch.tensor(trans_m, dtype=None), weight
