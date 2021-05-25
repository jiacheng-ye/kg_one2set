import logging

import torch
from torch.utils.data import DataLoader

from pykp.utils.io import KeyphraseDataset


def load_vocab(opt):
    # load vocab
    logging.info("Loading vocab from disk: %s" % opt.vocab)
    vocab = torch.load(opt.vocab + '/vocab.pt', 'wb')
    # assign vocab to opt
    opt.vocab = vocab
    logging.info('#(vocab)=%d' % len(vocab["word2idx"]))
    logging.info('#(vocab used)=%d' % opt.vocab_size)

    return vocab


def build_data_loader(data, opt, shuffle=True, load_train=True):
    keyphrase_dataset = KeyphraseDataset.build(examples=data, opt=opt, load_train=load_train)
    if not opt.one2many:
        collect_fn = keyphrase_dataset.collate_fn_one2one
    elif opt.fix_kp_num_len:
        collect_fn = keyphrase_dataset.collate_fn_fixed_tgt
    else:
        collect_fn = keyphrase_dataset.collate_fn_one2seq

    data_loader = DataLoader(dataset=keyphrase_dataset, collate_fn=collect_fn, num_workers=opt.batch_workers,
                             batch_size=opt.batch_size, shuffle=shuffle)
    return data_loader


def load_data_and_vocab(opt, load_train=True):
    vocab = load_vocab(opt)

    # constructor data loader
    logging.info("Loading train and validate data from '%s'" % opt.data)
    if opt.one2many:
        data_path = opt.data + '/%s.one2many.pt'
    else:
        data_path = opt.data + '/%s.one2one.pt'

    if load_train:
        # load training dataset
        train_data = torch.load(data_path % "train", 'wb')
        train_loader = build_data_loader(data=train_data, opt=opt, shuffle=True, load_train=True)
        logging.info('#(train data size: #(batch)=%d' % (len(train_loader)))

        # load validation dataset
        valid_data = torch.load(data_path % "valid", 'wb')
        valid_loader = build_data_loader(data=valid_data,  opt=opt, shuffle=False, load_train=True)
        logging.info('#(valid data size: #(batch)=%d' % (len(valid_loader)))
        return train_loader, valid_loader, vocab
    else:
        test_data = torch.load(data_path % "test", 'wb')
        test_loader = build_data_loader(data=test_data, opt=opt, shuffle=False, load_train=False)
        logging.info('#(test data size: #(batch)=%d' % (len(test_loader)))
        return test_loader, vocab
