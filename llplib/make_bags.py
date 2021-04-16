import numpy as np
from numpy.random import dirichlet
from sklearn.utils import shuffle
import random


class InsufficientDataPoints(Exception):
	pass


class InvalidAlpha(Exception):
	pass


def make_bags_dirichlet(train_y, num_class, bag_size, num_bags, alpha):
	if len(alpha) != num_class:
		raise InvalidAlpha("the dirichlet distribution's parameter should have length equal to num_class")

	lp_arr = (dirichlet(alpha, num_bags) * bag_size).astype(np.int)
	lp2counts = {}
	for row in range(lp_arr.shape[0]):
		lp = tuple(lp_arr[row])
		if lp not in lp2counts.keys():
			lp2counts[lp] = 0
		lp2counts[lp] += 1
	return _make_bags_counts(train_y, num_class, lp2counts)


def make_bags_counts(train_y, num_class, lp2counts):
	return _make_bags_counts(train_y, num_class, lp2counts)


def _make_bags_counts(train_y, num_class, lp2counts):
	train_y = np.array(train_y, dtype=np.int)  # y has to be integers starting from 0

	# first need to verify the number of data points
	total_label_counts = {}
	for label in range(num_class):
		total_label_counts[int(label)] = (train_y == label).astype(int).sum()
	expected_label_counts = {}
	for counts, num in lp2counts.items():
		for i in range(len(counts)):
			if i not in expected_label_counts.keys():
				expected_label_counts[i] = 0
			expected_label_counts[i] += counts[i] * num
	for label in range(num_class):
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
	bag_idx = 0
	for counts, num in lp2counts.items():
		for i in range(num):
			bag2indices[bag_idx] = []
			for label in range(num_class):
				class_indices = random.sample(label2indices[label], counts[label])
				label2indices[label] -= set(class_indices)
				for idx in class_indices:
					bag2indices[bag_idx].append(idx)
			bag2size[bag_idx] = len(bag2indices[bag_idx])
			bag2prop[bag_idx] = np.zeros((num_class,))
			for j in range(num_class):
				bag2prop[bag_idx][j] = np.sum(train_y[bag2indices[bag_idx]] == j) / bag2size[bag_idx]
			bag_idx += 1
	return bag2indices, bag2size, bag2prop


def make_bags_uniform(train_y, num_class, bag_size, num_bags):
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
		bag2prop[i] = np.zeros((num_class,))
		for j in range(num_class):
			bag2prop[i][j] = np.sum(train_y[bag2indices[i]] == j) / bag2size[i]
	return bag2indices, bag2size, bag2prop
