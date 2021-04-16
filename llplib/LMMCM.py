import numpy as np
from numpy.random import shuffle

import random
import itertools
import copy
from math import ceil

def one_inf_norm(matrix):
    return np.max(np.abs(matrix).sum(axis=0))


def make_groups(num_class, bag2indices, bag2size, bag2prop, assign_groups='random_greedy', group_weights='optimal_bound', **kwargs):
    if assign_groups == 'dummy':
        groups, group2lambda_m, group2bag, obj = _dummy_random_groups(num_class, bag2indices, bag2size, bag2prop)
    elif assign_groups == 'random_greedy':
        groups, group2lambda_m, group2bag, obj = _random_greedy(num_class, bag2size, bag2prop,
                                                                num_group_in_subset=kwargs['num_group_in_subset'],
                                                                max_combs=kwargs['max_combs'])
    else:
        raise ValueError("Unknow group assignment method")

    num_groups = len(groups)
    group2weight = dict()
    weight_sum = 0
    for i in range(num_groups):
        if group_weights == 'optimal_bound':
            group2weight[i] = np.divide(1, one_inf_norm(group2lambda_m[i].T) ** 2)
        elif group_weights == 'uniform':
            group2weight[i] = 1
        weight_sum += group2weight[i]
    for i in range(num_groups):
        group2weight[i] /= weight_sum

    return groups, group2weight, group2lambda_m, group2bag, obj


def make_groups_forward(num_class, bag2indices, bag2size, bag2prop):
    bag_ids = list(bag2indices.keys())
    num_groups = len(bag_ids) // num_class
    assert num_groups * num_class == len(bag_ids)
    random.shuffle(bag_ids)
    group2bag = {i: bag_ids[i * num_class:(i + 1) * num_class] for i in range(num_groups)}
    groups = list(group2bag.keys())

    num_groups = len(groups)

    group2transition = dict()
    group2gamma = dict()
    for group_id in groups:
        clean_prior = np.zeros((num_class, ))
        noisy_prior = np.zeros((num_class,))
        gamma_m = np.zeros((num_class, num_class))
        bags = group2bag[group_id]
        for row_idx in range(num_class):
            gamma_m[row_idx, :] = bag2prop[bags[row_idx]]
            noisy_prior[row_idx] = bag2size[bags[row_idx]]
            clean_prior += bag2prop[bags[row_idx]] * bag2size[bags[row_idx]]
        clean_prior /= clean_prior.sum()
        noisy_prior /= noisy_prior.sum()
        group2transition[group_id] = np.zeros((num_class, num_class))
        for i in range(num_class):
            for j in range(num_class):
                if clean_prior[j] != 0:
                    group2transition[group_id][i, j] = gamma_m[i, j] * noisy_prior[i] / clean_prior[j]
                else:
                    group2transition[group_id][i, j] = 0
        group2gamma[group_id] = gamma_m

    instance2group = {instance_id: group_id for group_id in groups for bag_id in group2bag[group_id] for instance_id in
                      bag2indices[bag_id]}

    noisy_y = np.zeros((sum([len(instances) for instances in bag2indices.values()]), ))
    for group_id in groups:
        for noisy_class, bag_id in enumerate(group2bag[group_id]):
            for instance_id in bag2indices[bag_id]:
                noisy_y[instance_id] = noisy_class

    return instance2group, group2gamma, group2transition, noisy_y


def get_mask(num_data, bag2size, bag2indices, groups, group2weight, group2lambda_m, group2bag):
    class_num = group2lambda_m[groups[0]].shape[0]
    mask = np.zeros((num_data, class_num))
    for group_id in groups:
        group_weight = group2weight[group_id]
        lambda_m = group2lambda_m[group_id]
        for c in range(class_num):
            bag_id = group2bag[group_id][c]
            bag_size = bag2size[bag_id]
            for instance_idx in bag2indices[bag_id]:
                mask[instance_idx, :] = lambda_m[c, :] * group_weight / (bag_size * class_num)
    return mask


def _random_greedy(num_class, bag2size, bag2prop, num_group_in_subset, max_combs):
    bag_ids = list(bag2prop.keys())
    num_bags = len(bag_ids)
    num_groups = num_bags // num_class
    group_ids = [i for i in range(num_groups)]

    obj = 0
    group2lambda_m = {}
    group2bag = {}

    bag_indices = copy.copy(bag_ids[:num_groups*num_class])
    shuffle(bag_indices)
    bag_indices_subsets = [set(bag_indices[num_class * i * num_group_in_subset: num_class * num_group_in_subset * (i+1)]) if
                            num_group_in_subset * (i+1) < num_groups else
                            set(bag_indices[num_class * i * num_group_in_subset:]) for i in range(ceil(num_groups / num_group_in_subset))]
    group_indices_subsets = [set(group_ids[i*num_group_in_subset: (i+1)*num_group_in_subset]) if
                              num_group_in_subset * (i+1) < num_groups else
                              set(group_ids[i*num_group_in_subset:]) for i in range(ceil(num_groups / num_group_in_subset))]
    for i in range(ceil(num_groups / num_group_in_subset)):
        local_obj, local_group2lambda_m, local_group2bag = _local_greedy(group_indices_subsets[i], bag_indices_subsets[i],
                                                                         num_class, bag2size, bag2prop, max_combs=max_combs)
        obj += local_obj
        group2lambda_m.update(local_group2lambda_m)
        group2bag.update(local_group2bag)
    return group_ids, group2lambda_m, group2bag, obj


def _local_greedy(group_ids, bag_ids, num_class, bag2size, bag2prop, max_combs=-1):
    num_bags = len(bag_ids)
    num_groups = len(group_ids)
    assert (num_groups * num_class == num_bags)

    estimated_obj = 0
    group2lambda_m = {}
    group2bag = {}

    bag_indices = copy.copy(bag_ids)
    for group_id in group_ids:
        # use a subset of all combinations
        if max_combs == -1:
            combs = itertools.combinations(bag_ids, num_class)
        else:
            combs = [random.sample(bag_indices, num_class) for i in range(max_combs)]

        local_optimal = np.NINF
        optimal_combs = None
        optimal_gamma = None
        # iterate through combs
        for bag_combs in combs:
            gamma_matrix = np.zeros((num_class, num_class))
            h = 0
            for idx in range(num_class):
                gamma_matrix[idx, :] = bag2prop[bag_combs[idx]]
                h += 1 / bag2size[bag_combs[idx]]

            local_obj = 1 / (one_inf_norm(np.linalg.inv(gamma_matrix)) ** 2 * h)
            if local_obj > local_optimal:
                local_optimal = local_obj
                optimal_combs = bag_combs
                optimal_gamma = gamma_matrix
        # find the optimal for s single S_i
        estimated_obj += local_optimal
        group2lambda_m[group_id] = np.linalg.inv(optimal_gamma.T)
        group2bag[group_id] = optimal_combs
        # update combs and bag_indices
        bag_indices -= set(optimal_combs)
    return estimated_obj, group2lambda_m, group2bag


def _dummy_random_groups(num_class, bag2indices, bag2size, bag2prop):
    bag_ids = list(bag2indices.keys())
    num_groups = len(bag_ids) // num_class
    groups = []
    group2bag = dict()
    obj = 0

    random.shuffle(bag_ids)
    for i in range(num_groups):
        group2bag[i] = bag_ids[i * num_class:(i + 1) * num_class]

    group2lambda_m = dict()
    for i in range(num_groups):
        gamma_i = np.zeros((num_class, num_class))
        bags = group2bag[i]
        for row_idx in range(num_class):
            gamma_i[row_idx, :] = bag2prop[bags[row_idx]]
        try:
            lambda_m = np.linalg.inv(gamma_i.T)
        except np.linalg.LinAlgError:
            print("singular gamma", gamma_i)
            return
        groups.append(i)
        group2lambda_m[i] = lambda_m

        h = 0
        for idx in range(num_class):
            h += 1 / bag2size[bags[idx]]
        obj += 1 / (one_inf_norm(np.linalg.inv(gamma_i)) ** 2 * h)
    return groups, group2lambda_m, group2bag, obj