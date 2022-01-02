import torch
import torch.nn as nn
from llpvatlib.train_fun import llpvat_train_by_bag
from llpfclib.train_fun import test_model
from llpvatlib.utils import VATLoss


def loss_f_test(x, y, device, epsilon=1e-8):
    x = torch.clamp(x, epsilon, 1 - epsilon)
    return nn.functional.nll_loss(torch.log(x), y, reduction='sum')


def llpvat(kl_train_dataset, scheduler, model, optimizer, test_loader, device, args, logger, json_data):
    train_loader = torch.utils.data.DataLoader(dataset=kl_train_dataset, batch_size=args.train_batch_size, shuffle=True)
    vat_loss_f = VATLoss(xi=args.vat_xi, eps=args.vat_eps, ip=args.vat_ip).to(device)
    for epoch in range(args.total_epochs):
        logger.info(f"Epoch-{epoch}")
        logger.info(f"      lr: {optimizer.param_groups[0]['lr']}")
        llpvat_train_by_bag(model, optimizer, train_loader, vat_loss_f, epoch, device, scheduler, logger)
        if test_loader is not None:
            acc, test_error = test_model(model, test_loader, loss_f_test, device)
            logger.info(f"        test_error = {test_error}, accuracy = {100 * acc}%")
            json_data['epoch_vs_test_accuracy'].append({'epoch': epoch, 'test_acc': acc, 'test_error': test_error})
