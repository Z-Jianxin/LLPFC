import argparse
from llpfclib.make_bags import make_bags_dirichlet, InsufficientDataPoints, make_bags_uniform, truncate_data
import torchvision
import pickle
import os
import random  # set random seed
import numpy as np  # set random seed


class InvalidArguments(Exception):
    pass


def get_args():
    parser = argparse.ArgumentParser(description="Partition data into bags for LLP")
    # required:
    parser.add_argument("-d", "--dataset", nargs='?', choices=["cifar10", "svhn", "emnist_letters"], required=True,
                        help="name of the dataset, the program uses torchvision.datasets")  # ToDo: add more data sets later
    parser.add_argument("-c", "--num_classes", nargs='?', type=int, required=True, metavar="10",
                        help="number of classes")
    parser.add_argument("-s", "--data_save_name", nargs='?', required=True, metavar="cifar10_1024_0",
                        help="name of the file to save")

    # optional:
    parser.add_argument("-l", "--data_folder_labeled", nargs='?', default="../data/labeled_data/",
                        metavar="../data/labeled_data/",
                        help="path to the folder of original data, if not exists, the dataset will be downloaded")
    parser.add_argument("-p", "--data_folder_llp", nargs='?', default="../data/llp_data/", metavar="../data/llp_data/",
                        help="path to save the training data for llp")
    parser.add_argument("-m", "--method", nargs='?', default="dirichlet", choices=["dirichlet", "uniform"],
                        help="method to generate bags")  # dirichlet, uniform
    parser.add_argument("-a", "--alpha", nargs="?", default="equal", choices=["equal"],
                        help="parameter of dirichlet distribution; required if use dirichlet to generate bags")  # Todo: add more to this
    parser.add_argument("-b", "--bag_size", nargs='?', type=int, metavar="1024",
                        help="size of bag; note not all bag sizes will equal to this number if use dirichlet")
    parser.add_argument("-n", "--num_bags", nargs='?', type=int, metavar="100",
                        help="number of bags to generate; it too large, the dataset may have insufficient data points")
    parser.add_argument("-r", "--seed", nargs='?', type=int, metavar="0", help="seed for all RNG")  # both random and numpy.random will use this seed
    return parser.parse_args()


def main(args):
    if args.dataset == "cifar10":
        train_dataset = torchvision.datasets.CIFAR10(root=args.data_folder_labeled, train=True, download=True)
        labels = train_dataset.targets
    elif args.dataset == "svhn":
        train_dataset = torchvision.datasets.SVHN(root=args.data_folder_labeled, split="train", download=True)
        labels = train_dataset.labels
    elif args.dataset == "emnist_letters":
        train_dataset = torchvision.datasets.EMNIST(root=args.data_folder_labeled, split="letters", train=True,
                                                    download=True)
        labels = train_dataset.targets - 1  # the labels range originally from 1 to 26
    else:
        raise InvalidArguments("Unknown dataset name: ", args.dataset)

    if args.method == "dirichlet":
        if args.alpha == "equal":
            alpha = tuple([1 for _ in range(args.num_classes)])
        else:
            raise InvalidArguments("Unknown choice of alpha: ", args.alpha)
        flag = 1
        fail_counter = 0
        while flag:
            try:
                bag2indices, bag2size, bag2prop = make_bags_dirichlet(labels, num_classes=args.num_classes,
                                                                      bag_size=args.bag_size, num_bags=args.num_bags,
                                                                      alpha=alpha)
                flag = 0
            except InsufficientDataPoints:
                flag = 1
                fail_counter += 1
                if fail_counter >= 100:
                    raise InsufficientDataPoints("THE DATA GENERATION PROCESS FAILS FOR 100 TIMES CONSECUTIVELY. "
                                                 "PLEASE CHECK ARGUMENTS OF --alpha %s, --bag_size %d, --num_bags %d"
                                                 % (args.alpha, args.bag_size, args.num_bags))
                continue
    elif args.method == "uniform":
        bag2indices, bag2size, bag2prop = make_bags_uniform(train_dataset.targets, args.num_classes, args.bag_size,
                                                            args.num_bags)
    else:
        raise InvalidArguments("Unknown method to generate bags: ", args.method)

    print("%d of bags generated, each bag has size %d, the random seed is %d, data is saved as %s" %
          (len(bag2indices.keys()), len(bag2indices[0]), args.seed, args.data_save_name))

    training_data, bag2indices = truncate_data(train_dataset.data, bag2indices)
    to_save = [training_data, bag2indices, bag2size, bag2prop]

    with open(os.path.join(args.data_folder_llp, args.data_save_name), 'wb') as f:
        pickle.dump(to_save, f)


if __name__ == "__main__":
    args = get_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    main(args)
