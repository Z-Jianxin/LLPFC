import argparse
import pickle
import random  # set the random seed for torchvision
import numpy as np
import torch

from models.NIN import NIN
from models.WideRes import wide_resnet_d_w
from models.ResNet import resnet18
from models.vgg import vgg19_bn, vgg16_bn
from models.densenet import densenet121
from models.LLPGAN_GEN import LLPGAN_GEN_MNIST, LLPGAN_GEN_COLOR
from llpfclib.utils import FORWARD_CORRECT_MNIST, FORWARD_CORRECT_CIFAR10, FORWARD_CORRECT_SVHN
from kllib.utils import KL_CIFAR10, KL_SVHN, KL_EMNIST

import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


class InvalidArguments(Exception):
    pass


def get_args():
    parser = argparse.ArgumentParser(description="train a model on LLP data")
    # required:
    parser.add_argument("-d",
                        "--dataset",
                        nargs='?',
                        choices=["cifar10", "svhn", "emnist_letters"],
                        required=True,
                        help="name of the dataset, the program uses torchvision.datasets")
    parser.add_argument("-p",
                        "--path_lp",
                        nargs='?',
                        required=True,
                        help="path to the label proportion dataset generated by make_data.py")
    parser.add_argument("-c", "--num_classes", nargs='?', type=int, required=True, help="number of classes")
    parser.add_argument("-f",
                        "--data_folder_labeled",
                        nargs='?',
                        required=True,
                        help="path to the folder of labeled test data, if not exists, the dataset will be downloaded")
    parser.add_argument("-log", "--logging_filename", nargs='?', required=True, help="path to save the log file")

    # optional:
    parser.add_argument("-a",
                        "--algorithm",
                        nargs='?',
                        choices=["llpfc", "kl", "llpvat", "llpgan"],
                        default="llpfc",
                        help="choose a training algorithm")
    parser.add_argument("-n",
                        "--network",
                        nargs='?',
                        choices=["wide_resnet_d_w", "nin", "ResNet18", "vgg19_bn", "vgg16_bn", "densenet121"],
                        default="wide_resnet_d_w",
                        help="the neural network model")
    parser.add_argument("-wrnd", "--WideResNet_depth", nargs='?', type=int, default=28)
    parser.add_argument("-wrnw", "--WideResNet_width", nargs='?', type=int, default=2)
    parser.add_argument("-dr",
                        "--drop_rate",
                        nargs="?",
                        type=float,
                        default=0.3,
                        help="the drop rate in dropout layers, for wide resnet")  # add more to this
    parser.add_argument("-o",
                        "--optimizer",
                        nargs="?",
                        default="Adamax",
                        choices=["Adamax", "LBFGS", "Adagrad", "nesterov", "AdamW", "SGD"],
                        help="optimizer of the neural network")
    parser.add_argument("-ams",
                        "--amsgrad",
                        nargs="?",
                        type=int,
                        default=0,
                        choices=[0, 1],
                        help="whether to use the AMSGrad variant of this algorithm")
    parser.add_argument("-l", "--lr", nargs="?", type=float, default=1e-3, help="learning rate")
    parser.add_argument("-m", "--momentum", nargs="?", type=float, default=0.9, help="momentum")
    parser.add_argument("-wd", "--weight_decay", nargs="?", type=float, default=0, help="weight decay")
    parser.add_argument("-e",
                        "--total_epochs",
                        nargs="?",
                        type=int,
                        default=200,
                        help="total number of epochs to train")
    parser.add_argument("-r",
                        "--num_epoch_regroup",
                        nargs="?",
                        type=int,
                        default=20,
                        help="groups will be regenerated every this number of epochs, " 
                             "only effective if the algorithm is llpfc")
    parser.add_argument("-np",
                        "--noisy_prior_choice",
                        nargs="?",
                        type=str,
                        default="approx",
                        choices=["approx", "uniform", "merge"],
                        help="the heuristics to estimate the noisy prior for each group, "
                             "approx solves the constrained optimization and uniform assigns uniform noisy priors")
    parser.add_argument("-v",
                        "--validate",
                        nargs='?',
                        type=int,
                        default=0,
                        choices=[0, 1],
                        help="if True, then validate on 10%% of the training data set; " 
                             "if False, output testing loss and accuracy while training"
                        )
    parser.add_argument("-b", "--train_batch_size", nargs='?', type=int, default=128, help="training batch size")
    parser.add_argument("-t", "--test_batch_size", nargs="?", type=int, default=256, help="test batch size")
    parser.add_argument("-s",
                        "--save_path",
                        nargs='?',
                        default=None,
                        help="path to save the trained model, model will not be saved if the path is None")
    parser.add_argument("-dv",
                        "--device",
                        nargs='?',
                        default="check",
                        choices=["cuda", "cpu", "check"],
                        help="device to train network; if it's check, use cuda whenever it's available")
    parser.add_argument("-w",
                        "--weights",
                        nargs='?',
                        choices=["uniform", "ch_vol"],
                        default="uniform",
                        help="set the weights for each group in llpfc")
    parser.add_argument("-sc",
                        "--scheduler",
                        nargs='?',
                        choices=["drop", "CAWR"],
                        default="drop",
                        help="set the scheduler of training lr")
    parser.add_argument("-ms",
                        "--milestones",
                        nargs='+',
                        type=int,
                        default=[],
                        help="number of epochs to drop lr if --scheduler is set to be 'drop'")
    parser.add_argument("-ga",
                        "--gamma",
                        nargs='?',
                        type=float,
                        default=0.1,
                        help="drop the learning rate by this factor if --scheduler is set to be 'drop'")
    parser.add_argument("-T0",
                        "--T_0",
                        nargs='?',
                        type=int,
                        default=10,
                        help="parameter of the CAWR scheduler")
    parser.add_argument("-Tm", "--T_mult", nargs='?', type=int, default=1, help="parameter of the CAWR scheduler")
    parser.add_argument("--seed", nargs='?', type=int, help="seed for all RNG")
    parser.add_argument("-fr",
                        "--full_reproducibility",
                        nargs='?',
                        type=int,
                        default=0,
                        choices=[0, 1],
                        help="choose to disable all nondeterministic algorithms, may at the cost of performance, "
                        " decrypted from now")
    # xi=1e-6, eps=6.0, ip=1
    parser.add_argument("-xi",
                        "--vat_xi",
                        nargs='?',
                        type=float,
                        default=1e-6,
                        help="parameter for vat loss, effective only algorithm=llpvat")
    parser.add_argument("-eps",
                        "--vat_eps",
                        nargs='?',
                        type=float,
                        default=6.0,
                        help="parameter for vat loss, effective only algorithm=llpvat")
    parser.add_argument("-ip",
                        "--vat_ip",
                        nargs='?',
                        type=float,
                        default=1,
                        help="parameter for vat loss, effective only algorithm=llpvat")
    parser.add_argument("-nd",
                        "--noise_dim",
                        nargs='?',
                        type=int,
                        default=500,
                        help="parameter for llpgan, the input dimension of the generator")
    parser.add_argument("-js",
                        "--path_to_json",
                        nargs='?',
                        type=str,
                        default=None,
                        help="will write the training results to this path if provided, write nothing if is none")
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
        image_size = 32
        in_channel = 3
    elif args.dataset == "svhn":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614)),  # mean-std of svhn
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614)),  # mean-std of svhn
        ])
        test_dataset = torchvision.datasets.SVHN(root=args.data_folder_labeled, split='test',
                                                 transform=transform_test, download=True)
        image_size = 32
        in_channel = 3
    elif args.dataset == "emnist_letters":
        image_size = 28
        in_channel = 1
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.1736, ), (0.3317, )),  # mean-std of emnist
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1736, ), (0.3317, )),  # mean-std of emnist
        ])
        if (args.network == "densenet121") or (len(args.network) >=3 and args.network[:3] == "vgg"):
            transform_train = transforms.Compose([
                transforms.Resize(32),  # resize the image for dense net as it has too many pool layers
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.1736,), (0.3317,)),  # mean-std of emnist
            ])
            transform_test = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize((0.1736,), (0.3317,)),  # mean-std of emnist
            ])
            image_size = 32
        test_dataset = torchvision.datasets.EMNIST(root=args.data_folder_labeled, split="letters", train=False,
                                                   transform=transform_test, download=True)
        test_dataset.targets = test_dataset.targets - 1  # the labels range originally from 1 to 26
    else:
        raise InvalidArguments("Unknown dataset name: ", args.dataset)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.test_batch_size, shuffle=False)

    num_classes = args.num_classes

    return_features = False
    if args.algorithm == "llpgan":
        return_features = True

    if args.network == "wide_resnet_d_w":
        model = wide_resnet_d_w(d=args.WideResNet_depth,
                                w=args.WideResNet_width,
                                dropout_rate=args.drop_rate,
                                num_classes=num_classes,
                                in_channel=in_channel,
                                image_size=image_size,
                                return_features=return_features
                                )
    elif args.network == "nin":
        model = NIN(num_classes=num_classes,
                    image_size=image_size,
                    in_channel=in_channel,
                    )
        if args.algorithm == "llpgan":
            raise InvalidArguments("NIN is not compatible with LLPGAN as it has no fully connected layer")
    elif args.network == "ResNet18":
        model = resnet18(num_classes, in_channel, return_features=return_features)
    elif args.network == "vgg19_bn":
        model = vgg19_bn(num_classes, in_channel, return_features=return_features)
    elif args.network == "vgg16_bn":
        model = vgg16_bn(num_classes, in_channel, return_features=return_features)
    elif args.network == "densenet121":
        model = densenet121(num_classes, in_channel, memory_efficient=False, return_features=return_features)
    else:
        raise InvalidArguments("Unknown selection of network: ", args.network)
    return llp_data, transform_train, num_classes, model, test_loader


def set_optimizer(args, model, total_epochs):
    if args.optimizer == "Adamax":
        optimizer = optim.Adamax(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08,
                                 weight_decay=args.weight_decay)
    elif args.optimizer == "Adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "LBFGS":
        optimizer = optim.LBFGS(model.parameters(), lr=args.lr)  # ToDo: implement the closure function and test
    elif args.optimizer == "nesterov":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,  weight_decay=args.weight_decay,
                              nesterov=True)
    elif args.optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
                              nesterov=False)
    elif args.optimizer == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08,
                                weight_decay=args.weight_decay, amsgrad=bool(args.amsgrad))
    else:
        raise InvalidArguments("Unknown selection of optimizer: ", args.optimizer)

    if args.scheduler == "drop":
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    elif args.scheduler == "CAWR":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.T_0, T_mult=args.T_mult)
    else:
        raise InvalidArguments("Unknown selection of scheduler: ", args.scheduler)
    return optimizer, scheduler


def set_dataset_class(args):
    competitors = ["kl", "llpvat", "llpgan"]
    if args.algorithm == "llpfc" and args.dataset == "cifar10":
        return FORWARD_CORRECT_CIFAR10
    elif args.algorithm == "llpfc" and args.dataset == "svhn":
        return FORWARD_CORRECT_SVHN
    elif args.algorithm == "llpfc" and args.dataset == "emnist_letters":
        return FORWARD_CORRECT_MNIST
    elif (args.algorithm in competitors) and args.dataset == "cifar10":
        return KL_CIFAR10
    elif (args.algorithm in competitors) and args.dataset == "svhn":
        return KL_SVHN
    elif (args.algorithm in competitors) and args.dataset == "emnist_letters":
        return KL_EMNIST
    raise InvalidArguments("Unknown combination of llp algorithm and dataset"
                           ": (%s, %s)" % (args.algorithm, args.dataset))


def set_generator(args):
    # return a tuple of (generator, optimizer of generator, )
    if args.dataset in ["cifar10", "svhn"]:
        return LLPGAN_GEN_COLOR(args.noise_dim)
    elif args.dataset == "emnist_letters":
        if (args.network == "densenet121") or (len(args.network) >= 3 and args.network[:3] == "vgg"):
            return LLPGAN_GEN_MNIST(args.noise_dim, 32, 32)
        return LLPGAN_GEN_MNIST(args.noise_dim, 28, 28)
    else:
        raise InvalidArguments("Unknown choice of dataset: %s" % args.dataset)
