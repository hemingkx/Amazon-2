import os
import torch

train_dir = '../dataset/train.csv'
test_dir = '../dataset/test.csv'

vocab_path = '../dataset/vocab.npz'

log_dir = '../log/train.log'
model_dir = '../model/best_model.bin'


embedding_dir = '../embedding/glove.6B/glove.6B.50d.txt'
pretrained_embedding = True


dev_split_size = 0.1      # 开发集比例
batch_size = 2048

embedding_size = 50
hidden_size = 256
lstm_layers = 2
lstm_drop_out = 0.2
nn_drop_out = 0
lr = 0.001
betas = (0.9, 0.999)
lr_step = 3
lr_gamma = 0.5
lengthen = True         # 是否使用变化RNN

epoch_num = 10
min_epoch_num = 3
patience = 0.0001
patience_num = 3

# gpu = '2'

