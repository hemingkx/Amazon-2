train_dir = './data/test.csv'
vocab_path = './data/vocab.npz'
log_dir = './log/train.log'
model_dir = './model/best_model.bin'
embedding_dir = '/home/xiaheming/workspace/Data/glove/glove.6B/glove.6B.50d.txt'

pretrained_embedding = False

split_size = 0.1
batch_size = 128

embedding_size = 50
hidden_size = 256
lstm_layers = 2
lstm_drop_out = 0.2
nn_drop_out = 0
lr = 0.001
betas = (0.9, 0.999)
lr_step = 3
lr_gamma = 0.5

epoch_num = 10
min_epoch_num = 3
patience = 0.0001
patience_num = 3

