import torch
import torch.nn as nn
from llpvatlib.train_fun import llpvat_train_by_bag
from llpfclib.train_fun import test_model


def loss_f_test(x, y, device, epsilon=1e-8):
    x = torch.clamp(x, epsilon, 1 - epsilon)
    return nn.functional.nll_loss(torch.log(x), y, reduction='sum')


def kl(model, optimizer, train_loader, alpha, consistency, scheduler, num_epochs, test_loader, device):
    for epoch in range(num_epochs):
        print(f"Epoch-{epoch}")
        print(f"        lr: {optimizer.param_groups[0]['lr']}")
        llpvat_train_by_bag(model, optimizer, train_loader, epoch, alpha, consistency, device, scheduler)
        acc, test_error = test_model(model, test_loader, loss_f_test, device)
        print(f"        accuracy = {100 * acc}%, test_error = {test_error}", flush=True)