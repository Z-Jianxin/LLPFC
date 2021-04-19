import pickle
import random # set the random seed for torchvision

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import numpy as np

from llplib.make_groups import make_groups_forward
from llplib.utils import FORWARD_CORRECT_MNIST, FORWARD_CORRECT_CIFAR10
from llplib.train_fun import train_model_forward_one_epoch, test_model
from models.NIN import NIN
from models.WideRes import wide_resnet28_2


class InvalidArguments(Exception):
    pass


def loss_f(x, y, weights, device, epsilon=1e-7):  # todo: Allow more choices of loss functions
    unweighted = nn.functional.nll_loss(torch.log(x + epsilon), y, reduction='none')
    weights /= weights.sum()
    return (unweighted * weights).sum()


def loss_f_test(x, y, device, epsilon=1e-7):
    return nn.functional.nll_loss(torch.log(x + epsilon), y, reduction='sum')


def llpfc(args):
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.random.manual_seed(args.seed)
        random.seed(args.seed)
    if args.full_reproducibility:
        # look https://pytorch.org/docs/stable/notes/randomness.html for references
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    num_classes = args.num_classes
    if args.device == "check":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print("using %s" % device)
    training_data, bag2indices, bag2size, bag2prop = pickle.load(open(args.path_lp, "rb"))

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

    if args.network == "wide_resnet_28_2":
        model = wide_resnet28_2(dropout_rate=args.drop_rate, num_classes=num_classes).to(device)
    elif args.network == "nin":
        model = NIN(num_classes=num_classes, image_size=image_size, in_channel=in_channel).to(device)
    else:
        raise InvalidArguments("Unknown selection of network: ", args.network)

    total_epochs = args.total_epochs
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    if args.optimizer == "Adamax":
        optimizer = torch.optim.Adamax(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08,
                                     weight_decay=args.weight_decay)
    elif args.optimizer == "Adagrad":
        optimizer = torch.optim.Adagrad(model.parameters())
    elif args.optimizer == "LBFGS":
        optimizer = torch.optim.LBFGS(model.parameters(), lr=args.lr)  # ToDo: implement the closure function and test
    elif args.optimizer == "nesterov":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True)
    else:
        raise InvalidArguments("Unknown selection of optimizer: ", args.optimizer)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=total_epochs//2, gamma=0.1)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.test_batch_size, shuffle=False)
    num_regroup = -1
    for epoch in range(total_epochs):
        if epoch % args.num_epoch_regroup == 0:
            flag = True
            while flag:
                try:
                    instance2group, group2transition, group2weights, noisy_y = \
                        make_groups_forward(num_classes, bag2indices, bag2size, bag2prop)
                    flag = False
                except np.linalg.LinAlgError:
                    flag = True
                    continue
            fc_train_dataset = FORWARD_CORRECT_CIFAR10(training_data, noisy_y, group2transition, group2weights,
                                                       instance2group, transform_train)
            llp_train_loader = torch.utils.data.DataLoader(dataset=fc_train_dataset, batch_size=args.train_batch_size,
                                                           shuffle=True)
            num_regroup += 1
        print(f"Regroup-{num_regroup} Epoch-{epoch}")
        print(f"    lr: {optimizer.param_groups[0]['lr']}")
        train_model_forward_one_epoch(model, loss_f, optimizer, llp_train_loader, epoch, device, scheduler=scheduler,
                                      constr=None)
        acc, test_error = test_model(model, test_loader, loss_f_test, device)
        print(f"accuracy = {100 * acc}%, test_error = {test_error}", flush=True)

    if args.save_path is not None:
        torch.save(model.state_dict(), args.save_path)
