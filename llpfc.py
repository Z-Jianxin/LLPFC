import torch
import torch.nn as nn

import numpy as np

from llpfclib.make_groups import make_groups_forward
from llpfclib.train_fun import train_model_forward_one_epoch, test_model


def loss_f(x, y, weights, device, epsilon=1e-7):  # todo: Allow more choices of loss functions
    unweighted = nn.functional.nll_loss(torch.log(x + epsilon), y, reduction='none')
    weights /= weights.sum()
    return (unweighted * weights).sum()


def loss_f_test(x, y, device, epsilon=1e-7):
    return nn.functional.nll_loss(torch.log(x + epsilon), y, reduction='sum')


def llpfc(num_classes, llp_data, transform_train, total_epochs, scheduler, model, optimizer, test_loader, dataset_class,
          weights, num_epoch_regroup, train_batch_size, device):
    training_data, bag2indices, bag2size, bag2prop = llp_data
    num_regroup = -1
    for epoch in range(total_epochs):
        if epoch % num_epoch_regroup == 0:
            flag = True
            while flag:
                try:
                    instance2group, group2transition, group2weights, noisy_y = \
                        make_groups_forward(num_classes, bag2indices, bag2size, bag2prop, weights=weights)
                    flag = False
                except np.linalg.LinAlgError:
                    flag = True
                    continue
            fc_train_dataset = dataset_class(training_data, noisy_y, group2transition, group2weights, instance2group,
                                             transform_train)
            llp_train_loader = torch.utils.data.DataLoader(dataset=fc_train_dataset, batch_size=train_batch_size,
                                                           shuffle=True)
            num_regroup += 1
        print(f"Regroup-{num_regroup} Epoch-{epoch}")
        print(f"    lr: {optimizer.param_groups[0]['lr']}")
        train_model_forward_one_epoch(model, loss_f, optimizer, llp_train_loader, device, epoch, scheduler=scheduler)
        acc, test_error = test_model(model, test_loader, loss_f_test, device)
        print(f"accuracy = {100 * acc}%, test_error = {test_error}", flush=True)
