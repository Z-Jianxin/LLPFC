import torch
from kllib.train_fun import compute_kl_loss_on_bagbatch
import numpy as np


def sigmoid_rampup(current, rampup_length):
    # modified from https://github.com/kevinorjohn/LLP-VAT/blob/a111d6785e8b0b79761c4d68c5b96288048594d6/llp_vat/
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def get_rampup_weight(weight, iteration, rampup):
    # modified from https://github.com/kevinorjohn/LLP-VAT/blob/a111d6785e8b0b79761c4d68c5b96288048594d6/llp_vat/
    alpha = weight * sigmoid_rampup(iteration, rampup)
    return alpha


def llp_loss_f(model, images, props, vat_loss_f, iteration, device):
    prop_loss = compute_kl_loss_on_bagbatch(model, images, props, device)
    alpha = get_rampup_weight(0.05, iteration, -1)  # hard-coded based on tsai and lin's implementation
    vat_loss = vat_loss_f(model,
                          torch.reshape(images, (-1, images.shape[-3], images.shape[-2], images.shape[-1])).to(device))
    return prop_loss, alpha, vat_loss


def llpvat_train_by_bag(model, optimizer, train_loader, vat_loss_f, epoch, device, scheduler, logger):
    model.train()
    total_step = len(train_loader)
    for i, (images, props) in enumerate(train_loader):
        prop_loss, alpha, vat_loss = llp_loss_f(model, images, props, vat_loss_f, i, device)
        loss = prop_loss + alpha * vat_loss
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % 100 == 0:
            logger.info('               Step [{}/{}], Loss: {:.4f}'.format(i + 1, total_step, loss.item()))
            logger.info('                             VAT Loss: {:.4f}'.format(vat_loss.item()))
            logger.info('                             KL Loss: {:.4f}'.format(prop_loss.item()))
            logger.info('                             alpha = {:.4f}'.format(alpha))
        if type(scheduler) == torch.optim.lr_scheduler.CosineAnnealingWarmRestarts:
            scheduler.step(epoch + i / total_step)
    if type(scheduler) == torch.optim.lr_scheduler.MultiStepLR:
        scheduler.step()