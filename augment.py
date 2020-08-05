import os
import torch
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
import utils
from configs.augment import AugmentConfig
from models.augment import AugmentCNN
from solvers.augment import train, validate


def main(config, writer, logger):
    logger.info("Logger is set - training augment start")

    # get data with meta info
    input_size, input_channels, n_classes, train_data, valid_data = utils.get_data(
        config.dataset, config.data_path, config.cutout_length, validation=True)

    criterion = nn.CrossEntropyLoss().cuda()
    use_aux = config.aux_weight > 0.
    model = AugmentCNN(input_size, input_channels, config.init_channels, n_classes, config.layers,
                       use_aux, config.genotype).cuda()

    # model size
    mb_params = utils.param_size(model)
    logger.info("Model size = {:.3f} MB".format(mb_params))

    # weights optimizer
    optimizer = torch.optim.SGD(model.parameters(), config.lr, momentum=config.momentum,
                                weight_decay=config.weight_decay)

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=config.batch_size,
                                               shuffle=True,
                                               num_workers=config.workers,
                                               pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_data,
                                               batch_size=config.batch_size,
                                               shuffle=False,
                                               num_workers=config.workers,
                                               pin_memory=True)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs)

    best_top1 = 0.
    # training loop
    for epoch in range(config.epochs):
        lr_scheduler.step()
        drop_prob = config.drop_path_prob * epoch / config.epochs
        model.drop_path_prob(drop_prob)

        # training
        train(train_loader, model, optimizer, criterion, epoch, config, writer, logger)

        # validation
        cur_step = (epoch+1) * len(train_loader)
        top1 = validate(valid_loader, model, criterion, epoch, cur_step, config, writer, logger)

        # save
        if best_top1 < top1:
            best_top1 = top1
            is_best = True
        else:
            is_best = False
        utils.save_checkpoint(model, config.path, is_best)

        print("")

    logger.info("Final best Prec@1 = {:.4%}".format(best_top1))


if __name__ == "__main__":
    
    config = AugmentConfig()
    
    # set gpu id
    if torch.cuda.is_available():
        if len(config.gpu) > 1:
            raise NotImplementedError('This implementation supports single GPU.')
    else:
        raise NotImplementedError('This implementation supports CPU.')


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
