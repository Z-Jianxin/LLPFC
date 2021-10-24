import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import mse_loss
from torch.autograd import Variable


def test_llpgan(model, test_loader, criterion, device):
    # test a model with fully label dataset
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        total_loss = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs, _ = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            prob = nn.functional.softmax(outputs, dim=1)
            loss = criterion(prob, labels, device)
            total_loss += loss.item()
    return correct / total, total_loss / total


def compute_dis_loss(dis, true_images, fake_images, props, device, lambd=1, epsilon=1e-8):
    # Forward pass
    batch_size, bag_size, channel, height, width = true_images.shape
    true_images = true_images.reshape((batch_size * bag_size, channel, height, width))
    true_outputs, _ = dis(true_images)
    fake_outputs, _ = dis(fake_images)

    # compute the lower bound of kl
    prob = nn.functional.softmax(true_outputs, dim=-1).reshape((batch_size, bag_size, -1))
    clamped_prob = torch.clamp(prob, epsilon, 1 - epsilon)
    log_prob = torch.log(clamped_prob)
    avg_log_prop = torch.mean(log_prob, dim=1)
    lower_kl_loss = -torch.sum(-props * avg_log_prop, dim=-1).mean() * lambd

    # compute the true/fake binary loss
    true_outputs_cat = torch.cat((true_outputs, torch.zeros(true_outputs.shape[0], 1).to(device)), dim=1)
    true_prob = 1 - nn.functional.softmax(true_outputs_cat, dim=1)[:, -1]
    clamped_true_prob = torch.clamp(true_prob, epsilon, 1 - epsilon)
    log_true_prob = torch.log(clamped_true_prob)
    avg_log_true_prop = -torch.mean(log_true_prob)

    fake_outputs_cat = torch.cat((fake_outputs, torch.zeros(fake_outputs.shape[0], 1).to(device)), dim=1)
    fake_prob = nn.functional.softmax(fake_outputs_cat, dim=1)[:, -1]
    clamped_fake_prob = torch.clamp(fake_prob, epsilon, 1 - epsilon)
    log_fake_prob = torch.log(clamped_fake_prob)
    avg_log_fake_prop = -torch.mean(log_fake_prob)
    return lower_kl_loss + avg_log_true_prop + avg_log_fake_prop


def compute_gen_loss(dis, true_images, fake_images):
    batch_size, bag_size, channel, height, width = true_images.shape
    true_images = true_images.reshape((batch_size * bag_size, channel, height, width))
    true_outputs, true_features = dis(true_images)
    fake_outputs, fake_features = dis(fake_images)
    loss = mse_loss(fake_features, true_features)
    return loss  # also return feature_maps to compute generator loss


def llpgan_train_by_bag(gen,
                        dis,
                        gen_opt,
                        dis_opt,
                        dis_sch,
                        gen_sch,
                        noise_dim,
                        train_loader,
                        epoch,
                        device,
                        logger
                        ):
    gen.train()
    dis.train()
    total_step = len(train_loader)
    for i, (images, props) in enumerate(train_loader):
        true_images = images.to(device)
        props = props.to(device)

        batch_data_points = true_images.shape[0] * true_images.shape[1]
        noise = Variable(torch.FloatTensor(np.random.normal(0, 1, (batch_data_points, noise_dim))).to(device))

        dis_opt.zero_grad()
        dis_loss = compute_dis_loss(dis, true_images, gen(noise).detach(), props, device, lambd=1)
        dis_loss.backward()
        dis_opt.step()

        gen_opt.zero_grad()
        gen_loss = compute_gen_loss(dis, true_images, gen(noise))
        gen_loss.backward()
        gen_opt.step()

        if (i + 1) % 100 == 0:
            logger.info('               Step [{}/{}], dis Loss: {:.4f}'.format(i + 1, total_step, dis_loss.item()))
            logger.info('                             gen Loss: {:.4f}'.format(gen_loss.item()))
        if type(dis_sch) == torch.optim.lr_scheduler.CosineAnnealingWarmRestarts:
            dis_sch.step(epoch + i / total_step)
        if type(gen_sch) == torch.optim.lr_scheduler.CosineAnnealingWarmRestarts:
            gen_sch.step(epoch + i / total_step)
    if type(dis_sch) == torch.optim.lr_scheduler.MultiStepLR:
        dis_sch.step()
    if type(gen_sch) == torch.optim.lr_scheduler.MultiStepLR:
        gen_sch.step()
