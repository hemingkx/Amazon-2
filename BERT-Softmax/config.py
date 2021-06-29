import os
import torch

data_dir = os.getcwd() + '/data/'
train_dir = data_dir + 'train.npz'
test_dir = data_dir + 'test.npz'
files = ['train', 'test']
bert_model = 'pretrained_bert_models/bert-base-uncased/'
model_dir = os.getcwd() + '/experiments/'
log_dir = model_dir + 'train.log'

# 最大长度
max_len = 196

# 训练集、验证集划分比例
dev_split_size = 0.1

# 是否加载训练好的NER模型
load_before = False

# 是否对整个BERT进行fine tuning
full_fine_tuning = True

# hyper-parameter
learning_rate = 3e-5
weight_decay = 0.01
clip_grad = 5

batch_size = 32
epoch_num = 5
min_epoch_num = 5
patience = 0.001
patience_num = 5

gpu = '2'

if gpu != '':
    device = torch.device(f"cuda:{gpu}")
else:
    device = torch.device("cpu")


if __name__ == "__main__":
    import torch
    a = torch.empty(2, 3, 4)
    a = a[:, 0, :]
    print(a.size())
