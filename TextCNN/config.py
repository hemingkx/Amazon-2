import os
import torch

data_dir = os.getcwd() + '/data/'
train = 'train.csv'
test = 'test.csv'
exp_dir = os.getcwd() + '/experiments/'
model_dir = exp_dir + 'model.pth'
log_dir = exp_dir + 'train.log'
glove = 'glove.6B.100d.txt'

max_vocab_size = 1000000

n_split = 10
train_split_size = 0.8       # 训练集比例
batch_size = 256
n_filters = 100          # 卷积 filter 的数量
filter_sizes = [2, 3, 4]    # filter 的尺寸
nn_drop_out = 0
lr = 0.001
betas = (0.9, 0.999)
lr_step = 3
lr_gamma = 0.5

epoch_num = 30
min_epoch_num = 8
patience = 0.0001
patience_num = 8

