import torch
from PIL import Image
from torch.utils.data import Sampler
from random import shuffle


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
	return x_truncated, y_truncated, instance2group_new


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
		self.data, self.noisy_y, self.instance2group = truncate_data_group(data, noisy_y, instance2group)
		self.group2transition = group2transition
		self.group2weights = group2weights
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


class GroupSampler(Sampler):

	def __init__(self, instance2group):
		self.group2instances = {}
		for inst, grp in instance2group.items():
			if grp not in self.group2instances.keys():
				self.group2instances[grp] = []
			self.group2instances[grp].append(inst)
		self.groups = list(self.group2instances.keys())
		self.MAX_BATCH_SIZE = 2048
		self.length = 0
		for group in self.groups:
			instances = self.group2instances[group]
			if len(instances) <= self.MAX_BATCH_SIZE:
				self.length += 1
			else:
				num_batches = len(instances) // self.MAX_BATCH_SIZE
				self.length += num_batches

	def __len__(self):
		return self.length

	def __iter__(self):
		shuffle(self.groups)
		for group in self.groups:
			instances = self.group2instances[group]
			shuffle(instances)
			if len(instances) <= self.MAX_BATCH_SIZE:
				yield instances
			else:
				num_batches = len(instances) // self.MAX_BATCH_SIZE
				for i in range(num_batches):
					start = self.MAX_BATCH_SIZE * i
					end = min(self.MAX_BATCH_SIZE * (i+1), len(instances))
					yield instances[start: end]
