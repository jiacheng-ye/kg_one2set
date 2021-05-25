import argparse
import os
import time
import torch

import config
from inference.evaluate import evaluate_greedy_generator
from pykp.model import Seq2SeqModel
from pykp.utils.io import build_interactive_predict_dataset
from utils.data_loader import load_vocab, build_data_loader
from utils.functions import common_process_opt, read_tokenized_src_file
from utils.functions import time_since


def process_opt(opt):
    opt = common_process_opt(opt)

    if not os.path.exists(opt.pred_path):
        os.makedirs(opt.pred_path)

    if torch.cuda.is_available():
        if not opt.gpuid:
            opt.gpuid = 0
        opt.device = torch.device("cuda:%d" % opt.gpuid)
    else:
        opt.device = torch.device("cpu")
        opt.gpuid = -1
        print("CUDA is not available, fall back to CPU.")

    return opt


def init_pretrained_model(opt):
    model = Seq2SeqModel(opt)
    model.load_state_dict(torch.load(opt.model))
    model.to(opt.device)
    model.eval()
    return model


def predict(test_data_loader, model, opt):
    if opt.fix_kp_num_len:
        from inference.set_generator import SetGenerator
        generator = SetGenerator.from_opt(model, opt)
    else:
        from inference.sequence_generator import SequenceGenerator
        generator = SequenceGenerator.from_opt(model, opt)
    evaluate_greedy_generator(test_data_loader, generator, opt)


def main(opt):
    vocab = load_vocab(opt)
    src_file = opt.src_file
    tokenized_src = read_tokenized_src_file(src_file, remove_title_eos=opt.remove_title_eos)

    if opt.one2many:
        mode = 'one2many'
    else:
        mode = 'one2one'

    test_data = build_interactive_predict_dataset(tokenized_src, opt, mode=mode, include_original=True)

    torch.save(test_data, open(opt.exp_path + "/test_%s.pt" % mode, 'wb'))

    test_loader = build_data_loader(data=test_data, opt=opt, shuffle=False, load_train=False)
    logging.info('#(test data size: #(batch)=%d' % (len(test_loader)))

    # init the pretrained model
    model = init_pretrained_model(opt)

    # Print out predict path
    logging.info("Prediction path: %s" % opt.pred_path)

    # predict the keyphrases of the src file and output it to opt.pred_path/predictions.txt
    start_time = time.time()
    predict(test_loader, model, opt)
    training_time = time_since(start_time)
    logging.info('Time for training: %.1f' % training_time)


if __name__ == '__main__':
    # load settings for training
    parser = argparse.ArgumentParser(
        description='interactive_predict.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    config.vocab_opts(parser)
    config.model_opts(parser)
    config.predict_opts(parser)
    opt = parser.parse_args()

    opt = process_opt(opt)
    logging = config.init_logging(log_file=opt.exp_path + '/output.log', stdout=True)
    logging.info('Parameters:')
    [logging.info('%s    :    %s' % (k, str(v))) for k, v in opt.__dict__.items()]

    main(opt)
