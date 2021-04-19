import numpy as np
from numpy.random import dirichlet, multinomial
from sklearn.utils import shuffle
import random


class InsufficientDataPoints(Exception):
	pass


class InvalidAlpha(Exception):
	pass


def make_bags_dirichlet(train_y, num_classes, bag_size, num_bags, alpha):
	if len(alpha) != num_classes:
		raise InvalidAlpha("the dirichlet distribution's parameter should have length equal to num_classes")

	multinomial_param = dirichlet(alpha, num_bags)
	bag_arr = np.zeros(multinomial_param.shape)
	for row_num in range(bag_arr.shape[0]):
		bag_arr[row_num, :] = multinomial(bag_size, multinomial_param[row_num, :])
	bag_arr = bag_arr.astype(np.int32)
	return _make_bags_counts(train_y, num_classes, bag_arr)


def make_bags_counts(train_y, num_classes, bag_arr):
	return _make_bags_counts(train_y, num_classes, bag_arr)


def _make_bags_counts(train_y, num_classes, lp_arr):
	train_y = np.array(train_y, dtype=np.int)  # y has to be integers starting from 0

	# first need to verify the number of data points
	total_label_counts = {}
	for label in range(num_classes):
		total_label_counts[int(label)] = (train_y == label).astype(int).sum()
	expected_label_counts = {i: np.sum(lp_arr[:, i]) for i in range(num_classes)}
	for label in range(num_classes):
		if total_label_counts[label] < expected_label_counts[label]:
			raise InsufficientDataPoints("Requested data points > total number of data points")
	# done checking

	label2indices = {}
	for i in range(len(train_y)):
		label = int(train_y[i])
		if label not in label2indices.keys():
			label2indices[label] = set({})
		label2indices[label].add(i)

	bag2indices, bag2size, bag2prop = {}, {}, {}
	for bag_idx in range(lp_arr.shape[0]):
		bag2indices[bag_idx] = []
		for label in range(num_classes):
			class_indices = random.sample(label2indices[label], lp_arr[bag_idx, label])
			label2indices[label] -= set(class_indices)
			bag2indices[bag_idx].extend(class_indices)
		bag2size[bag_idx] = len(bag2indices[bag_idx])
		bag2prop[bag_idx] = np.zeros((num_classes,))
		for j in range(num_classes):
			bag2prop[bag_idx][j] = np.sum(train_y[bag2indices[bag_idx]] == j) / bag2size[bag_idx]
	return bag2indices, bag2size, bag2prop


def make_bags_uniform(train_y, num_classes, bag_size, num_bags):
	train_y = np.array(train_y, dtype=np.int)  # y has to be integers starting from 0
	train_size = num_bags * bag_size
	train_y = train_y[:train_size]
	train_indices = shuffle(np.arange(0, train_size))
	bag2indices = dict()
	bag2size = dict()
	bag2prop = dict()
	for i in range(num_bags):
		bag2indices[i] = train_indices[i * bag_size:(i + 1) * bag_size]
		bag2size[i] = bag_size
		bag2prop[i] = np.zeros((num_classes,))
		for j in range(num_classes):
			bag2prop[i][j] = np.sum(train_y[bag2indices[i]] == j) / bag2size[i]
	return bag2indices, bag2size, bag2prop


def truncate_data(data, bag2indices):
	idx_list = []
	for bag_id in bag2indices.keys():
		idx_list.extend(bag2indices[bag_id])
	idx_list.sort()
	data_truncated = data[idx_list]
	idx2new = {idx_list[i]: i for i in range(len(idx_list))}
	bag2new = {bag_id: list(map(idx2new.get, bag2indices[bag_id])) for bag_id in bag2indices.keys()}
	return data_truncated, bag2new
