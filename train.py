import argparse
import json
import logging
import os
import time

import torch
from torch.optim import Adam

import config
import train_ml
from pykp.model import Seq2SeqModel
from utils.data_loader import load_data_and_vocab
from utils.functions import common_process_opt
from utils.functions import time_since


def process_opt(opt):
    opt = common_process_opt(opt)

    if opt.set_loss and (not opt.fix_kp_num_len or not opt.one2many):
        raise ValueError("Set fix_kp_num_len and one2many when using set loss!")

    if not os.path.exists(opt.exp_path):
        os.makedirs(opt.exp_path)
    if not os.path.exists(opt.model_path):
        os.makedirs(opt.model_path)

    # dump the setting (opt) to disk in order to reuse easily
    if opt.train_from:
        opt = torch.load(
            open(os.path.join(opt.model_path, 'initial.config'), 'rb')
        )
    else:
        torch.save(opt, open(os.path.join(opt.model_path, 'initial.config'), 'wb'))

    if torch.cuda.is_available():
        if not opt.gpuid:
            opt.gpuid = 0
        opt.device = torch.device("cuda:%d" % opt.gpuid)
    else:
        opt.device = torch.device("cpu")
        opt.gpuid = -1
        print("CUDA is not available, fall back to CPU.")

    return opt


def init_optimizer(model, opt):
    """
    mask the PAD <pad> when computing loss, before we used weight matrix, but not handy for copy-model, change to ignore_index
    :param model:
    :param opt:
    :return:
    """
    optimizer = Adam(params=filter(lambda p: p.requires_grad, model.parameters()),
                     lr=opt.learning_rate, betas=(0.9, 0.998), eps=1e-09)
    return optimizer


def init_model(opt):
    logging.info('======================  Model Parameters  =========================')
    model = Seq2SeqModel(opt)

    logging.info(model)
    total_params = sum([param.nelement() for param in model.parameters()])
    logging.info('model parameters: %d, %.2fM' % (total_params, total_params * 4 / 1024 / 1024))

    if opt.train_from:
        logging.info("loading previous checkpoint from %s" % opt.train_from)
        model.load_state_dict(torch.load(opt.train_from))
    return model.to(opt.device)


def main(opt):
    start_time = time.time()
    train_data_loader, valid_data_loader, vocab = load_data_and_vocab(opt, load_train=True)
    load_data_time = time_since(start_time)
    logging.info('Time for loading the data: %.1f' % load_data_time)

    start_time = time.time()
    model = init_model(opt)
    optimizer = init_optimizer(model, opt)

    train_ml.train_model(model, optimizer, train_data_loader, valid_data_loader, opt)
    training_time = time_since(start_time)
    logging.info('Time for training: %.1f' % training_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train.py',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    config.vocab_opts(parser)
    config.model_opts(parser)
    config.train_opts(parser)
    opt = parser.parse_args()
    opt = process_opt(opt)

    logging = config.init_logging(log_file=opt.exp_path + '/output.log', stdout=True)
    logging.info('Parameters:')
    [logging.info('%s    :    %s' % (k, str(v))) for k, v in opt.__dict__.items()]

    main(opt)
