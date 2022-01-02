import torch
import torch.nn as nn
from torch.distributions.constraints import simplex
from torch.utils.data import SubsetRandomSampler

import numpy as np

from llpfclib.make_groups import make_groups_forward
from llpfclib.train_fun import train_model_forward_one_epoch, test_model, validate_model_forward


def loss_f(x, y, weights, device, epsilon=1e-8):
    assert torch.all(simplex.check(x))
    x = torch.clamp(x, epsilon, 1 - epsilon)
    unweighted = nn.functional.nll_loss(torch.log(x), y, reduction='none')
    weights /= weights.sum()
    return (unweighted * weights).sum()


def loss_f_val(x, y, weights, device, epsilon=1e-8):
    assert torch.all(simplex.check(x))
    x = torch.clamp(x, epsilon, 1 - epsilon)
    unweighted = nn.functional.nll_loss(torch.log(x), y, reduction='none')
    return (unweighted * weights).sum()


def loss_f_test(x, y, device, epsilon=1e-8):
    x = torch.clamp(x, epsilon, 1 - epsilon)
    return nn.functional.nll_loss(torch.log(x), y, reduction='sum')


def llpfc(llp_data,
          transform_train,
          scheduler,
          model,
          optimizer,
          test_loader,
          dataset_class,
          device,
          args,
          logger,
          json_data):
    training_data, bag2indices, bag2size, bag2prop = llp_data
    num_regroup = -1
    train_sampler = None
    valid_sampler = None
    llp_valid_loader = None

    for epoch in range(args.total_epochs):
        if epoch % args.num_epoch_regroup == 0:
            instance2group, group2transition, instance2weight, noisy_y = make_groups_forward(args.num_classes,
                                                                                             bag2indices,
                                                                                             bag2size,
                                                                                             bag2prop,
                                                                                             args.noisy_prior_choice,
                                                                                             args.weights,
                                                                                             logger)
            fc_train_dataset = dataset_class(training_data,
                                             noisy_y,
                                             group2transition,
                                             instance2weight,
                                             instance2group,
                                             transform_train)
            if (llp_valid_loader is None) and args.validate:  # always use the first group assigment to validate
                VAL_PROP = 0.1
                num_data_points = len(fc_train_dataset)
                split = int(np.floor(VAL_PROP * num_data_points))
                indices = list(range(num_data_points))
                np.random.shuffle(indices)
                train_indices, val_indices = indices[split:], indices[:split]
                train_sampler = SubsetRandomSampler(train_indices)
                valid_sampler = SubsetRandomSampler(val_indices)
                llp_valid_loader = torch.utils.data.DataLoader(dataset=fc_train_dataset, sampler=valid_sampler,
                                                               batch_size=args.train_batch_size)
            if train_sampler is None:
                llp_train_loader = torch.utils.data.DataLoader(dataset=fc_train_dataset, shuffle=True,
                                                               batch_size=args.train_batch_size)
            else:
                llp_train_loader = torch.utils.data.DataLoader(dataset=fc_train_dataset, sampler=train_sampler,
                                                               batch_size=args.train_batch_size)
            num_regroup += 1
        logger.info(f"Regroup-{num_regroup} Epoch-{epoch}")
        logger.info(f"		lr: {optimizer.param_groups[0]['lr']}")
        train_model_forward_one_epoch(model, loss_f, optimizer, llp_train_loader, device, epoch, scheduler, logger)
        if test_loader is not None:
            acc, test_error = test_model(model, test_loader, loss_f_test, device)
            logger.info(f"      test_error = {test_error}, accuracy = {100 * acc}%")
            json_data['epoch_vs_test_accuracy'].append({'epoch': epoch, 'test_acc': acc, 'test_error': test_error})
        if args.validate:
            assert llp_valid_loader is not None
            val_loss = validate_model_forward(model, loss_f_val, llp_valid_loader, device)
            logger.info(f"      valid_loss = {val_loss}")

