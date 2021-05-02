import torch
import torch.nn as nn
from llpvatlib.train_fun import llpvat_train_by_bag, validate_model_llpvat
from llpfclib.train_fun import test_model


def loss_f_test(x, y, device, epsilon=1e-8):
    x = torch.clamp(x, epsilon, 1 - epsilon)
    return nn.functional.nll_loss(torch.log(x), y, reduction='sum')


def kl(model, optimizer, train_loader, alpha, use_vat, scheduler, num_epochs, val_loader, test_loader, device,
       logger):
    for epoch in range(num_epochs):
        logger.info(f"Epoch-{epoch}")
        logger.info(f"      lr: {optimizer.param_groups[0]['lr']}")
        llpvat_train_by_bag(model, optimizer, train_loader, epoch, alpha, use_vat, device, scheduler, logger)
        if test_loader is not None:
            acc, test_error = test_model(model, test_loader, loss_f_test, device)
            logger.info(f"        test_error = {test_error}, accuracy = {100 * acc}%")
        if val_loader is not None:
            val_error = validate_model_llpvat(model, val_loader, device)
            logger.info(f"        val_error = {val_error}")
