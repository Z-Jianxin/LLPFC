import argparse
import pickle
import random  # set the random seed for torchvision
import numpy as np  # set the random seed for torchvision

import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from models.NIN import NIN
from models.WideRes import wide_resnet_d_w
from llpfclib.utils import FORWARD_CORRECT_MNIST, FORWARD_CORRECT_CIFAR10
from llpvatlib.utils import LLPVAT_CIFAR10
from llpfc import llpfc
from llpvat import kl


class InvalidArguments(Exception):
    pass


def get_args():
    parser = argparse.ArgumentParser(description="train a model on LLP data")
    # required:
    parser.add_argument("-d", "--dataset", nargs='?', choices=["cifar10"], required=True,
                        help="name of the dataset, the program uses torchvision.datasets")  # ToDo: add more data sets later
    parser.add_argument("-p", "--path_lp", nargs='?', required=True,
                        help="path to the label proportion dataset generated by make_data.py")
    parser.add_argument("-c", "--num_classes", nargs='?', type=int, required=True, help="number of classes")
    parser.add_argument("-f", "--data_folder_labeled", nargs='?', required=True,
                        help="path to the folder of labeled test data, if not exists, the dataset will be downloaded")

    # optional:
    parser.add_argument("-a", "--algorithm", nargs='?', choices=["llpfc", "kl"], default="llpfc",
                        help="choose a training algorithm")  # ToDo: add more after implementing competitors
    parser.add_argument("-n", "--network", nargs='?', choices=["wide_resnet_d_w", "nin"],
                        default="wide_resnet_d_w", help="the neural network model")  # ToDo: include more networks
    parser.add_argument("-wrnd", "--WideResNet_depth", nargs='?', type=int, default=28)
    parser.add_argument("-wrnw", "--WideResNet_width", nargs='?', type=int, default=2)
    parser.add_argument("-dr", "--drop_rate", nargs="?", type=float, default=0.3,
                        help="the drop rate in dropout layers, for wide resnet")  # add more to this
    parser.add_argument("-o", "--optimizer", nargs="?", default="Adamax",
                        choices=["Adamax", "LBFGS", "Adagrad", "nesterov"],
                        help="optimizer of the neural network")
    parser.add_argument("-l", "--lr", nargs="?", type=float, default=1e-3, help="learning rate")
    parser.add_argument("-m", "--momentum", nargs="?", type=float, default=0.9, help="momentum")
    parser.add_argument("-wd", "--weight_decay", nargs="?", type=float, default=0, help="weight decay")
    parser.add_argument("-e", "--total_epochs", nargs="?", type=int, default=200,
                        help="total number of epochs to train")
    parser.add_argument("-r", "--num_epoch_regroup", nargs="?", type=int, default=20,
                        help="groups will be regenerated every this number of epochs, " 
                             "only effective if the algorithm is llpfc")
    parser.add_argument("-v", "--validate", nargs='?', type=bool, default=False,
                        help="if True, then validate on 10%% of the training data set; " 
                             "if False, output testing loss and accuracy will training")
    parser.add_argument("-b", "--train_batch_size", nargs='?', type=int, default=128, help="training batch size")
    parser.add_argument("-t", "--test_batch_size", nargs="?", type=int, default=256, help="test batch size")
    parser.add_argument("-s", "--save_path", nargs='?', default=None,
                        help="path to save the trained model, model will not be saved if the path is None")
    parser.add_argument("-dv", "--device", nargs='?', default="check", choices=["cuda", "cpu", "check"],
                        help="device to train network; if it's check, use cuda whenever it's available")
    parser.add_argument("-w", "--weights", nargs='?', choices=["uniform", "ch_vol"], default="uniform",
                        help="set the weights for each group in llpfc")
    parser.add_argument("-sc", "--scheduler", nargs='?', choices=["scheduler", "ReduceLROnPlateau", "CAWR"],
                        default="scheduler", help="set the scheduler of training lr")
    parser.add_argument("-T0", "--T_0", nargs='?', type=int, default=10, help="parameter of the CAWR scheduler")
    parser.add_argument("-Tm", "--T_mult", nargs='?', type=int, default=1, help="parameter of the CAWR scheduler")
    parser.add_argument("--seed", nargs='?', type=int, help="seed for all RNG")
    parser.add_argument("-fr", "--full_reproducibility", nargs='?', type=bool, default=False,
                        help="choose to disable all nondeterministic algorithms, may at the cost of performance")
    return parser.parse_args()


def set_reproducibility(args):
    # random seed:
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.random.manual_seed(args.seed)
        random.seed(args.seed)
    # use deterministic algorithms
    if args.full_reproducibility:
        # look https://pytorch.org/docs/stable/notes/randomness.html for references
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)


def set_device(args):
    # configure device
    if args.device == "check":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    return device


def set_data_and_model(args):
    # read the training data
    llp_data = pickle.load(open(args.path_lp, "rb"))

    if args.dataset == "cifar10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),  # mean-std of cifar10
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),  # mean-std of cifar10
        ])
        test_dataset = torchvision.datasets.CIFAR10(root=args.data_folder_labeled, train=False,
                                                    transform=transform_test, download=True)
        image_size = (32, 32)
        in_channel = 3
    else:
        raise InvalidArguments("Unknown dataset name: ", args.dataset)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.test_batch_size, shuffle=False)

    num_classes = args.num_classes
    if args.network == "wide_resnet_d_w":
        model = wide_resnet_d_w(d=args.WideResNet_depth, w=args.WideResNet_width, dropout_rate=args.drop_rate,
                                num_classes=num_classes)
    elif args.network == "nin":
        model = NIN(num_classes=num_classes, image_size=image_size, in_channel=in_channel)
    else:
        raise InvalidArguments("Unknown selection of network: ", args.network)
    return llp_data, transform_train, num_classes, model, test_loader


def set_optimizer(args, model, total_epochs):
    if args.optimizer == "Adamax":
        optimizer = optim.Adamax(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08,
                                 weight_decay=args.weight_decay)
    elif args.optimizer == "Adagrad":
        optimizer = optim.Adagrad(model.parameters())
    elif args.optimizer == "LBFGS":
        optimizer = optim.LBFGS(model.parameters(), lr=args.lr)  # ToDo: implement the closure function and test
    elif args.optimizer == "nesterov":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True)
    else:
        raise InvalidArguments("Unknown selection of optimizer: ", args.optimizer)

    if args.scheduler == "scheduler":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=total_epochs//2, gamma=0.1)
    elif args.scheduler == "ReduceLROnPlateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)
    elif args.scheduler == "CAWR":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.T_0, T_mult=args.T_mult)
    else:
        raise InvalidArguments("Unknown selection of scheduler: ", args.scheduler)
    return optimizer, scheduler


def set_dataset_class(args):
    if args.algorithm == "llpfc" and args.dataset == "cifar10":
        return FORWARD_CORRECT_CIFAR10
    elif args.algorithm == "kl" and args.dataset == "cifar10":
        return LLPVAT_CIFAR10
    raise InvalidArguments("Unknown llp algorithm: ", args.algorithm)


def main(args):
    set_reproducibility(args)
    device = set_device(args)
    print("using %s" % device)
    llp_data, transform_train, num_classes, model, test_loader = set_data_and_model(args)
    model = model.to(device)
    total_epochs = args.total_epochs
    optimizer, scheduler = set_optimizer(args, model, total_epochs)
    if args.algorithm == "llpfc":
        dataset_class = set_dataset_class(args)
        llpfc(num_classes, llp_data, transform_train, total_epochs, scheduler, model, optimizer, test_loader,
              dataset_class, args.weights, args.num_epoch_regroup, args.train_batch_size, device)
    elif args.algorithm == "kl":
        dataset_class = set_dataset_class(args)
        training_data, bag2indices, bag2size, bag2prop = llp_data
        llpvat_train_dataset = dataset_class(training_data, bag2indices, bag2prop, transform_train)
        train_loader = torch.utils.data.DataLoader(dataset=llpvat_train_dataset, batch_size=args.train_batch_size,
                                                   shuffle=True)
        alpha = 1.0
        consistency = None
        kl(model, optimizer, train_loader, alpha, consistency, scheduler, total_epochs, test_loader, device)


    if args.save_path is not None:
        torch.save(model.state_dict(), args.save_path)


if __name__ == "__main__":
    parser = get_args()
    main(parser)
