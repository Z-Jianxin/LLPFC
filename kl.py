import torch
import torch.nn as nn
from kllib.train_fun import kl_train_by_bag, validate_model_kl
from llpfclib.train_fun import test_model


def loss_f_test(x, y, device, epsilon=1e-8):
    x = torch.clamp(x, epsilon, 1 - epsilon)
    return nn.functional.nll_loss(torch.log(x), y, reduction='sum')


def kl(model, optimizer, train_loader, scheduler, num_epochs, val_loader, test_loader, device, logger, json_data):
    for epoch in range(num_epochs):
        logger.info(f"Epoch-{epoch}")
        logger.info(f"      lr: {optimizer.param_groups[0]['lr']}")
        kl_train_by_bag(model, optimizer, train_loader, epoch, device, scheduler, logger)
        if test_loader is not None:
            acc, test_error = test_model(model, test_loader, loss_f_test, device)
            logger.info(f"        test_error = {test_error}, accuracy = {100 * acc}%")
            json_data['epoch_vs_test_accuracy'].append({'epoch': epoch, 'test_acc': acc, 'test_error': test_error})
        if val_loader is not None:
            val_error = validate_model_kl(model, val_loader, device)
            logger.info(f"        val_error = {val_error}")
