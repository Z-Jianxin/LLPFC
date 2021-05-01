import numpy as np
import random
from scipy.spatial import ConvexHull
from scipy.special import factorial
from numpy.linalg import matrix_rank


class InvalidChoiceOfWeights(Exception):
    pass


def make_groups_forward(num_classes, bag2indices, bag2size, bag2prop, weights):
    bag_ids = list(bag2indices.keys())
    num_groups = len(bag_ids) // num_classes
    assert num_groups > 0
    random.shuffle(bag_ids)
    group2bag = {i: bag_ids[i * num_classes:(i + 1) * num_classes] for i in range(num_groups)}
    group2bag[-1] = bag_ids[num_groups * num_classes:]
    groups = list(group2bag.keys())

    group2transition = dict()
    group2gamma = dict()
    for group_id in groups:
        if group_id == -1:
            continue
        clean_prior = np.zeros((num_classes, ))
        noisy_prior = np.zeros((num_classes,))
        gamma_m = np.zeros((num_classes, num_classes))
        bags = group2bag[group_id]
        for row_idx in range(num_classes):
            gamma_m[row_idx, :] = bag2prop[bags[row_idx]]
            noisy_prior[row_idx] = bag2size[bags[row_idx]]
            clean_prior += bag2prop[bags[row_idx]] * bag2size[bags[row_idx]]
        clean_prior /= clean_prior.sum()
        noisy_prior /= noisy_prior.sum()
        group2transition[group_id] = np.zeros((num_classes, num_classes))
        for i in range(num_classes):
            for j in range(num_classes):
                if clean_prior[j] != 0:
                    group2transition[group_id][i, j] = gamma_m[i, j] * noisy_prior[i] / clean_prior[j]
                else:
                    group2transition[group_id][i, j] = 0
        if matrix_rank(group2transition[group_id]) != num_classes:  # todo: change the way sample bags and fix singular transition matrices
            print("singular transition")
            np.set_printoptions(precision=3)
            np.set_printoptions(suppress=True)
            print(group2transition[group_id])
        group2gamma[group_id] = gamma_m

    instance2group = {instance_id: group_id for group_id in groups for bag_id in group2bag[group_id] for instance_id in
                      bag2indices[bag_id]}
    # calculate the weights of groups
    group2weights = {}
    if weights == "ch_vol":
        weights_sum = 0
        for group_id, trans_m in group2transition.items():
            simplex = np.vstack((np.transpose(trans_m), np.zeros(num_classes)))
            group2weights[group_id] = ConvexHull(simplex).volume
            weights_sum += group2weights[group_id]
        for group_id, trans_m in group2transition.items():
            group2weights[group_id] /= weights_sum
    elif weights == "uniform":
        group2weights = {group_id: 1.0 for group_id, trans_m in group2transition.items()}
    else:
        raise InvalidChoiceOfWeights("unknown way to determine weights %s, use either ch_vol or uniform" % weights)

    # set the noisy labels
    noisy_y = -np.ones((sum([len(instances) for instances in bag2indices.values()]), ))
    for group_id in groups:
        if group_id == -1:
            continue
        for noisy_class, bag_id in enumerate(group2bag[group_id]):
            for instance_id in bag2indices[bag_id]:
                noisy_y[instance_id] = noisy_class

    return instance2group, group2transition, group2weights, noisy_y
