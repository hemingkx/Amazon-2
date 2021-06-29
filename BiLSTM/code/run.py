import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR

import os

import config
import logging
import numpy as np
from model import BiLSTM
from embedding import embedding

from data_loader import EEDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from train import train, val
from sklearn.model_selection import train_test_split

import utils
from utils import load_dataset
from vocabulary import Vocabulary

def run():
    """train without k-fold"""
    # set the logger
    utils.set_logger(config.log_dir)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 设置gpu为命令行参数指定的id
    # if config.gpu != '':
    #     device = torch.device(f"cuda:{config.gpu}")
    # else:
    #     device = torch.device("cpu")
    
    logging.info("device: {}".format(device))
    # 处理数据，分离文本和标签

    sentences_train, labels_train = load_dataset(config.train_dir)
    sentences_test, labels_test = load_dataset(config.test_dir)

    # for local debug:
    sentences, labels = load_dataset(config.test_dir)
    sentences_train, labels_train = sentences[:20], labels[:20]
    sentences_test, labels_test = sentences[100:110], labels[100:110]
    config.dev_split_size = 0.5
    config.patience_num = 2
    config.min_epoch_num = 2

    # 建立词表
    vocab = Vocabulary(config)
    vocab.get_vocab(sentences_train, sentences_test)

    # 分离出验证集
    sentences_train, sentences_dev, labels_train, labels_dev = train_test_split(sentences_train, labels_train, test_size=config.dev_split_size, random_state=0)

    sentences_train.reset_index(drop=True, inplace=True)
    sentences_dev.reset_index(drop=True, inplace=True)
    sentences_test.reset_index(drop=True, inplace=True)
    labels_train.reset_index(drop=True, inplace=True)
    labels_dev.reset_index(drop=True, inplace=True)
    labels_test.reset_index(drop=True, inplace=True)


    # build dataset
    train_dataset = EEDataset(sentences_train, labels_train, vocab)
    dev_dataset = EEDataset(sentences_dev, labels_dev, vocab)
    test_dataset = EEDataset(sentences_test, labels_test, vocab)
    # build data_loader
    

    # build data_loader
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=False, collate_fn=train_dataset.collate_fn)

    dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size,
                            shuffle=True, collate_fn=dev_dataset.collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                             shuffle=False, collate_fn=test_dataset.collate_fn)

    # get pretrained embedding
    if config.pretrained_embedding:
        embedding_weight = embedding(vocab)
    else:
        embedding_weight = None

    # model
    model = BiLSTM(embedding_size=config.embedding_size,
                       hidden_size=config.hidden_size,
                       vocab_size=vocab.vocab_size(),
                       target_size=1,
                       num_layers=config.lstm_layers,
                       lstm_drop_out=config.lstm_drop_out,
                       nn_drop_out=config.nn_drop_out,
                       pretrained_embedding=config.pretrained_embedding,
                       embedding_weight=embedding_weight,
                       lengthen = config.lengthen)
    model.to(device)
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=config.betas)
    scheduler = StepLR(optimizer, step_size=config.lr_step, gamma=config.lr_gamma)
    
    # train and test
    train(train_loader, dev_loader, vocab, model, optimizer, scheduler, device)
    with torch.no_grad():
        # test on the final test set
        test_accuracy, test_loss = val(test_loader, vocab, model, device)
    
    logging.info("test accuracy: {}, test loss: {}".format(test_accuracy, test_loss))

    return test_loss, test_accuracy


if __name__ == '__main__':
    # if os.path.exists(config.log_dir):
    #     os.remove(config.log_dir)
    result = run()
    # k_fold_run()
