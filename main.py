import argparse


def get_args():
    parser = argparse.ArgumentParser(description="train a model on LLP data")
    # required:
    parser.add_argument("--dataset", nargs='?', choices=["cifar10"], required=True,
                        help="name of the dataset, the program uses torchvision.datasets")  # add more data sets later
    parser.add_argument("--dataset", nargs='?', )
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
