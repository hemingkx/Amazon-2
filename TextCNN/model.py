import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):

    def __init__(self, embedding_size, vocab_size, target_size, n_filters, filter_sizes, nn_drop_out):
        super(TextCNN, self).__init__()
        self.nn_drop_out = nn_drop_out
        # nn.Embedding: parameter size (num_words, embedding_dim)
        # for every word id, output a embedding for this word
        # input size: N x W, N is batch size, W is max sentence len
        # output size: (N, W, embedding_dim), embedding all the words
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        # 下面这是一个【并联】的卷积
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels = 1, out_channels = n_filters, kernel_size = (fs, embedding_size)) for fs in filter_sizes
        ])
        if nn_drop_out > 0:
            self.dropout = nn.Dropout(nn_drop_out)
        self.classifier = nn.Linear(n_filters * len(filter_sizes), target_size) # 每个filter可以算出最后一个值，然后是并联结果，最后做concat

    def forward(self, unigrams, input_tags, training=True):
        if len(unigrams.size()) < 2:
            unigrams.unsqueeze_(0)
            input_tags.unsqueeze_(0)
        uni_embeddings = self.embedding(unigrams)    # [batch_size, sent_len, emb_dim]
        uni_embeddings = uni_embeddings.unsqueeze(1)  # [batch_size, 1, sent_len, emb_dim] 多出来的这个1是channels
        conved = [F.relu(conv(uni_embeddings)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        sequence_output = torch.cat(pooled, dim = 1) # 得到整句话的表示 [batch_size, n_filters * len(filter_sizes)]
        if training and self.nn_drop_out > 0:
            sequence_output = self.dropout(sequence_output)
        class_output = self.classifier(sequence_output) 
        tag_scores = torch.sigmoid(class_output)
        input_tags.unsqueeze_(1)
        CE_loss = nn.BCELoss(reduction = "sum")(tag_scores.double(), input_tags.double())# 这地方求和，没有取平均数    
        return tag_scores, CE_loss
