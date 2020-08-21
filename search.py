import os
import torch
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
import utils
from configs.search import SearchConfig
from models.search import SearchCNNController
from architect import Architect
from visualize import plot
from solvers.search import train, validate


def main(config, writer, logger):
    logger.info("Logger is set - training search start")

    input_size, input_channels, n_classes, train_data = utils.get_data(
        config.dataset, config.data_path, cutout_length=0, validation=False)

    net_crit = nn.CrossEntropyLoss().to(config.device)
    model = SearchCNNController(input_channels, config.init_channels, n_classes, config.layers, net_crit).to(config.device)

    # weights optimizer
    w_optim = torch.optim.SGD(model.weights(), config.w_lr, momentum=config.w_momentum,
                                    weight_decay=config.w_weight_decay)
    # alphas optimizer
    alpha_optim = torch.optim.Adam(model.alphas(), config.alpha_lr, betas=(config.alpha_beta1, config.alpha_beta2),
                                    weight_decay=config.alpha_weight_decay)

    # split data to train/validation
    n_train = len(train_data)
    split = n_train // 2
    indices = list(range(n_train))
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])
    train_loader = torch.utils.data.DataLoader(train_data,
                                                batch_size=config.batch_size,
                                                sampler=train_sampler,
                                                num_workers=config.workers,
                                                pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(train_data,
                                                batch_size=config.batch_size,
                                                sampler=valid_sampler,
                                                num_workers=config.workers,
                                                pin_memory=True)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(w_optim, config.epochs, eta_min=config.w_lr_min)
    architect = Architect(model, config.w_momentum, config.w_weight_decay)

    # training loop
    best_top1 = 0.
    for epoch in range(config.epochs):
        lr_scheduler.step()
        lr = lr_scheduler.get_lr()[0]

        model.print_alphas(logger)

        # training
        train(train_loader, valid_loader, model, architect, w_optim, alpha_optim, lr, epoch, config, writer, logger)

        # validation
        cur_step = (epoch+1) * len(train_loader)
        top1 = validate(valid_loader, model, epoch, cur_step, config, writer, logger)

        # log
        # genotype
        genotype = model.genotype()
        logger.info("genotype = {}".format(genotype))

        # genotype as a image
        plot_path = os.path.join(config.plot_path, "EP{:02d}".format(epoch+1))
        caption = "Epoch {}".format(epoch+1)
        plot(genotype.normal, plot_path + "-normal", caption)
        plot(genotype.reduce, plot_path + "-reduce", caption)

        # save
        if best_top1 < top1:
            best_top1 = top1
            best_genotype = genotype
            is_best = True
        else:
            is_best = False
        utils.save_checkpoint(model, config.path, is_best)

    logger.info("Final best Prec@1 = {:.4%}".format(best_top1))
    logger.info("Best Genotype = {}".format(best_genotype))


if __name__ == "__main__":

    config = SearchConfig()

    # set gpu id
    if torch.cuda.is_available():
        if len(config.gpus) > 1:
            raise NotImplementedError('This implementation only supports single GPU.')
        config.device = torch.device("cuda:{}".format(config.gpus))
    else:
        config.device = torch.device("cpu")

    # set tensorboard
    writer = SummaryWriter(log_dir=os.path.join(config.path, "tb"))
    writer.add_text('config', config.as_markdown(), 0)

    # set logger
    logger = utils.get_logger(os.path.join(config.path, "{}.log".format(config.name)))
    config.print_params(logger.info)

    # set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark = True

    main(config, writer, logger)