import torch
import torch.nn as nn
from llpganlib.train_fun import llpgan_train_by_bag, test_llpgan


def loss_f_test(x, y, device, epsilon=1e-8):
    x = torch.clamp(x, epsilon, 1 - epsilon)
    return nn.functional.nll_loss(torch.log(x), y, reduction='sum')


def llpgan(kl_train_dataset, dis, gen, dis_opt, gen_opt, dis_sch, gen_sch, test_loader, device, args, logger):
    train_loader = torch.utils.data.DataLoader(dataset=kl_train_dataset, batch_size=args.train_batch_size, shuffle=True)
    for epoch in range(args.total_epochs):
        logger.info(f"Epoch-{epoch}")
        logger.info(f"      dis lr: {dis_opt.param_groups[0]['lr']}")
        logger.info(f"      gen lr: {gen_opt.param_groups[0]['lr']}")
        llpgan_train_by_bag(gen,
                            dis,
                            gen_opt,
                            dis_opt,
                            dis_sch,
                            gen_sch,
                            args.noise_dim,
                            train_loader,
                            epoch,
                            device,
                            logger)
        if test_loader is not None:
            acc, test_error = test_llpgan(dis, test_loader, loss_f_test, device)
            logger.info(f"        test_error = {test_error}, accuracy = {100 * acc}%")
