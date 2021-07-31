import os
import torch
import logging
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split

import config
from model import BiLSTM
from embedding import embedding
from data_loader import MyDataset
from train import train, val
from utils import load_dataset, set_logger
from vocabulary import Vocabulary


def run():
    """train without k-fold"""
    # set the logger
    set_logger(config.log_dir)
    device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
    # 处理数据，分离文本和标签
    sentences_train, labels_train = load_dataset(config.train_dir)
    # 缩小数据集
    sentences_t, sentences_train, labels_t, labels_train = train_test_split(sentences_train, labels_train,
                                                                            test_size=0.1,
                                                                            random_state=0)
    # 分离出验证集
    sentences_train, sentences_test, labels_train, labels_test = train_test_split(sentences_train, labels_train,
                                                                                  test_size=config.split_size,
                                                                                  random_state=0)
    # 建立词表
    vocab = Vocabulary(config)
    vocab.get_vocab(sentences_train, sentences_test)

    sentences_train, sentences_val, labels_train, labels_val = train_test_split(sentences_train, labels_train,
                                                                                test_size=config.split_size,
                                                                                random_state=0)

    # build dataset
    train_dataset = MyDataset(sentences_train, labels_train, vocab)
    dev_dataset = MyDataset(sentences_val, labels_val, vocab)
    test_dataset = MyDataset(sentences_test, labels_test, vocab)

    # build data_loader
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, collate_fn=train_dataset.collate_fn)

    dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size,
                            shuffle=False, collate_fn=dev_dataset.collate_fn)
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
                   embedding_weight=embedding_weight)
    model.to(device)
    # optimizer
    optimizer = Adam(model.parameters(), lr=config.lr, betas=config.betas)
    scheduler = StepLR(optimizer, step_size=config.lr_step, gamma=config.lr_gamma)

    # train and test
    train(train_loader, dev_loader, model, optimizer, scheduler, device)
    with torch.no_grad():
        # test on the final test set
        test_accuracy, test_loss = val(test_loader, model, device)

    logging.info("test accuracy: {}, test loss: {}".format(test_accuracy, test_loss))


if __name__ == '__main__':
    if os.path.exists(config.log_dir):
        os.remove(config.log_dir)
    run()
