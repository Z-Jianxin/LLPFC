import sys
import numpy as np  # set the random seed for torchvision
import logging
import torch

from torch.utils.data.sampler import SubsetRandomSampler

from llpfc import llpfc
from kl import kl
from llpvat import llpvat
from llpgan import llpgan
from utils import set_optimizer, set_device, set_reproducibility, set_data_and_model, set_dataset_class, get_args, set_generator


def main(args):
    set_reproducibility(args)
    logging.basicConfig(level=logging.INFO, filename=args.logging_filename, filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    logger = logging.getLogger()

    device = set_device(args)
    logger.info("\n\n")
    logger.info("program starts")
    logger.info("running arguments %s" % sys.argv)
    llp_data, transform_train, num_classes, model, test_loader = set_data_and_model(args)
    model = model.to(device)
    total_epochs = args.total_epochs
    optimizer, scheduler = set_optimizer(args, model, total_epochs)

    if args.algorithm == "llpfc":
        dataset_class = set_dataset_class(args)
        llpfc(llp_data, transform_train, scheduler, model, optimizer, test_loader, dataset_class, device, args, logger)
    elif args.algorithm == "kl":
        dataset_class = set_dataset_class(args)
        training_data, bag2indices, bag2size, bag2prop = llp_data
        kl_train_dataset = dataset_class(training_data, bag2indices, bag2prop, transform_train)

        train_loader = torch.utils.data.DataLoader(dataset=kl_train_dataset,
                                                   batch_size=args.train_batch_size,
                                                   shuffle=True)
        val_loader = None
        kl(model, optimizer, train_loader, scheduler, total_epochs, val_loader, test_loader, device, logger)
    elif args.algorithm == "llpvat":
        dataset_class = set_dataset_class(args)
        training_data, bag2indices, bag2size, bag2prop = llp_data
        kl_train_dataset = dataset_class(training_data, bag2indices, bag2prop, transform_train)
        llpvat(kl_train_dataset, scheduler, model, optimizer, test_loader, device, args, logger)
    elif args.algorithm == "llpgan":
        dataset_class = set_dataset_class(args)
        training_data, bag2indices, bag2size, bag2prop = llp_data
        kl_train_dataset = dataset_class(training_data, bag2indices, bag2prop, transform_train)
        gen = set_generator(args)
        gen = gen.to(device)
        gen_opt, gen_sch = set_optimizer(args, gen, total_epochs)
        llpgan(kl_train_dataset, model, gen, optimizer, gen_opt, scheduler, gen_sch, test_loader, device, args, logger)
    if args.save_path is not None:
        torch.save(model.state_dict(), args.save_path)
    logger.info("training completed")
    logger.info("")


if __name__ == "__main__":
    parser = get_args()
    main(parser)
