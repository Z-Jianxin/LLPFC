import numpy as np
import random
from scipy.spatial import ConvexHull
from scipy.special import factorial
from numpy.linalg import matrix_rank
from scipy.optimize import minimize, Bounds, LinearConstraint


class InvalidChoiceOfWeights(Exception):
    pass


class InvalidChoiceOfNoisyPrior(Exception):
    pass


def approx_noisy_prior(gamma_m, clean_prior):
    def ls_error(x, A, b):
        return 0.5 * np.sum((np.matmul(A, x) - b) ** 2)

    def grad(x, A, b):
        return np.matmul(np.matmul(np.transpose(A), A), x) - np.matmul(np.transpose(A), b)

    def hess(x, A, b):
        return np.matmul(np.transpose(A), A)

    x0 = np.random.rand(clean_prior.shape[0])
    x0 /= np.sum(x0)

    res = minimize(ls_error,
                   x0,
                   args=(np.transpose(gamma_m), clean_prior),
                   method='trust-constr',
                   jac=grad,
                   hess=hess,
                   bounds=Bounds(np.zeros(x0.shape), np.ones(x0.shape)),
                   constraints=LinearConstraint(np.ones(x0.shape), np.ones(1), np.ones(1)),
                   )
    return res.x


def make_a_group(num_classes, clean_prior, bag_ids, bag2prop, noisy_prior_choice, logger):
    bags_list = random.sample(bag_ids, num_classes)
    gamma_m = np.zeros((num_classes, num_classes))
    for row_idx in range(num_classes):
        gamma_m[row_idx, :] = bag2prop[bags_list[row_idx]]
    if noisy_prior_choice == 'approx':
        noisy_prior_approx = approx_noisy_prior(np.transpose(gamma_m), clean_prior)
    elif noisy_prior_choice == 'uniform':
        noisy_prior_approx = np.ones((num_classes,)) / num_classes
    else:
        raise InvalidChoiceOfNoisyPrior("Unknown choice of noisy prior: %s" % noisy_prior_choice)
    assert np.all(noisy_prior_approx >= 0)
    assert (np.sum(noisy_prior_approx) - 1) < 1e-4
    clean_prior_approx = np.matmul(np.transpose(gamma_m), noisy_prior_approx)

    transition_m = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            transition_m[i, j] = gamma_m[i, j] * noisy_prior_approx[i] / clean_prior_approx[j]  # clean_prior can't be 0 in this case

    if matrix_rank(transition_m) != num_classes:
        logger.warning("singular transition matrix")
    if np.any(noisy_prior_approx < 0):
        logger.warning("negative prior of noisy labels")
    return bags_list, noisy_prior_approx, transition_m


def _pow_normalize(x, t):
    """
    returns normalized x**t
    this function is used to control the probability of bag assignment
    """
    exp = x ** t
    return exp / np.sum(exp, axis=0)


def merge_bags(num_classes, bag2indices, bag2size, bag2prop, logger, t=10):
    assert len(bag2indices.keys()) >= num_classes

    # merge bags in mega-bags
    # make sure we have at least one bag in each mega-bag
    bag2mega = {max(bag2prop, key=lambda x: bag2prop[x][c]): c for c in range(num_classes)}
    for b in bag2prop:
        if b in bag2mega.keys():
            continue
        bag2mega[b] = np.random.choice(np.arange(0, num_classes), p=_pow_normalize(bag2prop[b], t))
    mega2prop = {}
    gamma_m = np.zeros((num_classes, num_classes))
    for c in range(num_classes):
        prop = np.zeros((num_classes, ))
        for b in bag2prop.keys():
            if bag2mega[b] == c:
                prop += bag2prop[b]
        prop = prop/np.sum(prop)
        mega2prop[c] = prop
        gamma_m[c, :] = prop

    # compute noisy transition matrix
    clean_prior = np.zeros((num_classes,))
    for bag_id in bag2size.keys():
        clean_prior += bag2prop[bag_id] * bag2size[bag_id]
    clean_prior /= np.sum(clean_prior)
    noisy_prior = np.matmul(np.linalg.inv(np.transpose(gamma_m)), clean_prior)

    transition_m = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            transition_m[i, j] = gamma_m[i, j] * noisy_prior[i] / clean_prior[j]  # clean_prior can't be 0 in this case
    if matrix_rank(transition_m) != num_classes:
        logger.warning("singular transition matrix")
    if np.any(noisy_prior < 0):
        logger.warning("negative prior of noisy labels")

    instance2group = {instance_id: 0 for bag_id in bag2indices.keys() for instance_id in bag2indices[bag_id]}
    noisy_y = -np.ones((sum([len(instances) for instances in bag2indices.values()]),))
    instance2weight = np.zeros((sum([len(instances) for instances in bag2indices.values()]),))
    group2transition = {}
    for bag_id in bag2indices.keys():
        for instance_id in bag2indices[bag_id]:
            noisy_y[instance_id] = bag2mega[bag_id]
            group2transition[instance_id] = transition_m
            instance2weight[instance_id] = noisy_prior[bag2mega[bag_id]]
    assert (noisy_y == -1).sum() == 0
    return instance2group, group2transition, instance2weight, noisy_y


def make_groups_forward(num_classes, bag2indices, bag2size, bag2prop, noisy_prior_choice, weights, logger):
    if noisy_prior_choice == "merge":
        return merge_bags(num_classes, bag2indices, bag2size, bag2prop, logger,)

    bag_ids = set(bag2indices.keys())
    num_groups = len(bag_ids) // num_classes
    assert num_groups > 0

    clean_prior = np.zeros((num_classes, ))
    for bag_id in bag2size.keys():
        clean_prior += bag2prop[bag_id] * bag2size[bag_id]
    clean_prior /= np.sum(clean_prior)

    group2bag = {}
    group2noisyp = {}
    group2transition = {}
    group_id = 0
    groups = []
    while len(bag_ids) >= num_classes:
        bags_list, noisy_prior, transition_m = make_a_group(num_classes,
                                                            clean_prior,
                                                            bag_ids,
                                                            bag2prop,
                                                            noisy_prior_choice,
                                                            logger)
        bag_ids = bag_ids - set(bags_list)
        group2bag[group_id], group2noisyp[group_id], group2transition[group_id] = bags_list, noisy_prior, transition_m
        groups.append(group_id)
        group_id += 1
    group2bag[-1] = list(bag_ids)  # bags that are not in a group
    groups.append(-1)

    instance2group = {instance_id: group_id for group_id in groups for bag_id in group2bag[group_id] for
                      instance_id in bag2indices[bag_id]}

    # calculate the weights of groups
    if weights == 'uniform':
        group2weights = {group_id: 1.0 for group_id, trans_m in group2transition.items()}
    else:
        raise InvalidChoiceOfWeights("Unknown way to determine weights %s, use either ch_vol or uniform" % weights)

    # set the noisy labels
    noisy_y = -np.ones((sum([len(instances) for instances in bag2indices.values()]),))
    instance2weight = np.zeros((sum([len(instances) for instances in bag2indices.values()]),))
    for group_id in groups:
        if group_id == -1:
            continue
        for noisy_class, bag_id in enumerate(group2bag[group_id]):
            for instance_id in bag2indices[bag_id]:
                noisy_y[instance_id] = noisy_class
                instance2weight[instance_id] = group2noisyp[group_id][noisy_class] * group2weights[group_id]

    return instance2group, group2transition, instance2weight, noisy_y
