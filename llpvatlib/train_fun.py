import torch
import torch.nn as nn


def compute_kl_loss_on_bagbatch(model, images, props, device, epsilon=1e-8):
    # Move tensors to the configured device
    images = images.to(device)
    props = props.to(device)
    # Forward pass
    batch_size, bag_size, channel, height, width = images.shape
    images = images.reshape((batch_size * bag_size, channel, height, width))
    outputs = model(images)
    prob = nn.functional.softmax(outputs, dim=-1).reshape((batch_size, bag_size, -1))
    avg_prob = torch.mean(prob, dim=1)
    avg_prob = torch.clamp(avg_prob, epsilon, 1 - epsilon)
    loss = torch.sum(-props * torch.log(avg_prob), dim=-1).mean()
    return loss


def validate_model_llpvat(model, val_loader, device):
    model.eval()
    total_loss = 0
    for i, (images, props) in enumerate(val_loader):
        total_loss += compute_kl_loss_on_bagbatch(model, images, props, device).item()
    return total_loss


def VATLoss():
    # ToDo: implement this
    return 0


def llpvat_train_by_bag(model, optimizer, train_loader, epoch, alpha, consistency, device, scheduler, logger):
    model.train()
    total_step = len(train_loader)
    for i, (images, props) in enumerate(train_loader):
        loss = alpha * compute_kl_loss_on_bagbatch(model, images, props, device) + VATLoss()
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % 100 == 0:
            logger.info('Step [{}/{}], Loss: {:.4f}'.format(i + 1, total_step, loss.item()))
        if type(scheduler) == torch.optim.lr_scheduler.CosineAnnealingWarmRestarts:
            scheduler.step(epoch + i / total_step)
    if type(scheduler) == torch.optim.lr_scheduler.StepLR:
        scheduler.step()
    elif type(scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
        scheduler.step(validate_model_llpvat(model, train_loader, device))
