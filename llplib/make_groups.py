import numpy as np
import random


def make_groups_forward(num_class, bag2indices, bag2size, bag2prop):
    bag_ids = list(bag2indices.keys())
    num_groups = len(bag_ids) // num_class
    assert num_groups * num_class == len(bag_ids)
    random.shuffle(bag_ids)
    group2bag = {i: bag_ids[i * num_class:(i + 1) * num_class] for i in range(num_groups)}
    groups = list(group2bag.keys())

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
