import argparse
from llplib.make_bags import make_bags_dirichlet, InsufficientDataPoints, make_bags_uniform
import torchvision
import pickle
import os
import random  # set random seed
import numpy as np  # set random seed


def get_args():
    parser = argparse.ArgumentParser(description="Partition data into bags for LLP")
    # required:
    parser.add_argument("--dataset", nargs='?', choices=["cifar10"], required=True,
                        help="name of the dataset, the program uses torchvision.datasets")  # add more data sets later
    parser.add_argument("--num_classes", nargs='?', type=int, required=True, metavar="10", help="number of classes")
    parser.add_argument("--data_save_name", nargs='?', required=True, metavar="cifar10_1024_0",
                        help="name of the file to save")

    # optional:
    parser.add_argument("--data_folder_labeled", nargs='?', default="../data/labeled_data/",
                        metavar="../data/labeled_data/",
                        help="path to the folder of original data, if not exists, the dataset will be downloaded")
    parser.add_argument("--data_folder_llp", nargs='?', default="../data/llp_data/", metavar="../data/llp_data/",
                        help="path to save the training data for llp")
    parser.add_argument("--method", nargs='?', default="dirichlet", choices=["dirichlet", "uniform"],
                        help="method to generate bags")  # dirichlet, uniform # Todo: implement "counts"
    parser.add_argument("--alpha", nargs="?", default="equal", choices=["equal"],
                        help="parameter of dirichlet distribution; required if use dirichlet to generate bags")  # add more to this
    parser.add_argument("--bag_size", nargs='?', type=int, metavar="1024",
                        help="size of bag; note not all bag sizes will equal to this number if use dirichlet")
    parser.add_argument("--num_bags", nargs='?', type=int, metavar="100",
                        help="number of bags to generate; it too large, the dataset may have insufficient data points")
    parser.add_argument("--seed", nargs='?', type=int, metavar="0",
                        help="seed of RNG for both random and numpy.random")  # both random and numpy.random will use this seed
    return parser.parse_args()


def main(args):
    if args.dataset == "cifar10":
        train_dataset = torchvision.datasets.CIFAR10(root=args.data_folder_labeled, train=True, download=True)

    if args.method == "dirichlet":
        if args.alpha == "equal":
            alpha = tuple([1 for _ in range(args.num_classes)])
        flag = 1
        fail_counter = 0
        while flag:
            try:
                bag2indices, bag2size, bag2prop = make_bags_dirichlet(train_dataset.targets, num_class=args.num_classes,
                                                                      bag_size=args.bag_size, num_bags=args.num_bags,
                                                                      alpha=alpha)
                flag = 0
            except InsufficientDataPoints:
                flag = 1
                fail_counter += 1
                if fail_counter >= 10:
                    print("THE DATA GENERATION PROCESS FAILS FOR 10 TIMES CONSECUTIVELY. PLEASE CHECK ARGUMENTS "
                          "OF --alpha, --bag_size, --num_bags")
                    raise InsufficientDataPoints
                continue
    elif args.method == "uniform":
        bag2indices, bag2size, bag2prop = make_bags_uniform(train_dataset.targets, args.num_classes, args.bag_size,
                                                            args.num_bags)

    to_save = [bag2indices, bag2size, bag2prop]

    with open(os.path.join(args.data_folder_llp, args.data_save_name), 'wb') as f:
        pickle.dump(to_save, f)


if __name__ == "__main__":
    args = get_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    main(args)