import torch
import random
from torch import optim
from torch.optim.lr_scheduler import StepLR

import os
from utils import set_logger
import config
import logging
import numpy as np
from model import TextCNN
from train import train, dev
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torchtext import data, vocab
from distributed_utils import is_main_process, cleanup, dist


def test(test_iterator, device):
    """test model performance on the final test set"""
    test_loader = DataLoader([(data.text, data.label) for data in test_iterator], batch_size=1, shuffle = False)
    # 载入模型
    model = torch.load(config.model_dir)
    model.to(device)
    logging.info("--------Load model from {}--------".format(config.model_dir))
    # 评价
    metric = dev(test_loader, model, device, mode = "test")
    acc = metric['acc']
    f1 = metric['f1']
    p = metric['p']
    r = metric['r']
    test_loss = metric['loss']
    logging.info("final test loss: {}, acc: {}, f1 score: {}, precision:{}, recall: {}".format(test_loss, acc, f1, p, r))

def run():
   
    # 制作数据集
    TEXT = data.Field(tokenize='spacy', tokenizer_language="en_core_web_sm", batch_first=True, lower=True)
    LABEL = data.LabelField(dtype = torch.float)
    datafields = [("label", LABEL), ("title", TEXT), ("text", TEXT)]
    train_data,test_data = data.TabularDataset.splits(
        path = config.data_dir, 
        train=config.train,   
        test=config.test,
        format = 'csv',
        fields = datafields
        )
    # 载入预训练词表
    vectors = vocab.Vectors(name = config.glove, cache=config.data_dir)
    # 从验证集中分理处开发集，因为要多卡，所以要把数据的随机数种子固定下来 
    train_data, valid_data = train_data.split(split_ratio = config.train_split_size, random_state=random.seed(0))
    # 建立词表 torch.Tensor.normal_方式初始化unk
    TEXT.build_vocab(train_data, max_size = config.max_vocab_size, vectors = vectors, unk_init = torch.Tensor.normal_)
    LABEL.build_vocab(train_data)
    vocab_size, embedding_size = TEXT.vocab.vectors.shape
    # build data_loader
    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits((train_data, valid_data, test_data),batch_size = config.batch_size, sort_key = lambda x:len(x.text))
    train_loader = DataLoader([(data.text, data.label) for data in train_iterator], batch_size=1, sampler=DistributedSampler(train_iterator), num_workers=4)
    dev_loader = DataLoader([(data.text, data.label) for data in valid_iterator], batch_size=1, sampler=DistributedSampler(valid_iterator), num_workers=4)
    # model
    model = TextCNN(embedding_size=embedding_size,
               vocab_size=vocab_size,
               target_size= 1,
               n_filters = config.n_filters,
               filter_sizes = config.filter_sizes,
               nn_drop_out=config.nn_drop_out)
    # 载入预训练词向量
    pretrained_embeddings = TEXT.vocab.vectors
    model.embedding.weight.data.copy_(pretrained_embeddings)
    model.to(device)
    
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=config.betas)
    scheduler = StepLR(optimizer, step_size=config.lr_step, gamma=config.lr_gamma)
    model = DistributedDataParallel(model, find_unused_parameters=True, device_ids=[local_rank], output_device=local_rank)
    
    # train and test
    train(train_loader, dev_loader, model, optimizer, scheduler, device)
    
    if is_main_process(): 
        # 使用主进程测试
        with torch.no_grad():
            test(test_iterator, device)  # 测试
    cleanup() # 结束所有进程
    

if __name__ == '__main__':
    set_logger(config.log_dir)
    # set up distributed device
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)
    print(f"local rank: {local_rank}, global rank: {rank}")
    
    run()
