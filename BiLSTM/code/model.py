import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class BiLSTM(nn.Module):

    def __init__(self, embedding_size, hidden_size, vocab_size, target_size, num_layers, lstm_drop_out, nn_drop_out, lengthen = False, pretrained_embedding=False, embedding_weight=None):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.nn_drop_out = nn_drop_out
        self.lengthen = lengthen
        # nn.Embedding: parameter size (num_words, embedding_dim)
        # for every word id, output a embedding for this word
        # input size: N x W, N is batch size, W is max sentence len
        # output size: (N, W, embedding_dim), embedding all the words
        # lengthen: 使用变长RNN: True; 使用定长RNN: False
        if pretrained_embedding:
            self.embedding = nn.Embedding.from_pretrained(embedding_weight)
            self.embedding.weight.requires_grad = True
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_size)
            
        self.bilstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=num_layers,
            dropout=lstm_drop_out if num_layers > 1 else 0,
            bidirectional=True
        )
        if nn_drop_out > 0:
            self.dropout = nn.Dropout(nn_drop_out)
        self.linear = nn.Linear(hidden_size * 2, target_size) 

    def forward(self, unigrams, input_tags, input_lens, training=True):
        uni_embeddings = self.embedding(unigrams)
        hidden = None
        if self.lengthen:
            # 选择变长RNN
            uni_embeddings_packed = pack_padded_sequence(uni_embeddings, input_lens, batch_first=True)
            _,(hidden, _) = self.bilstm(uni_embeddings_packed)
        else:
            # 选择定长RNN
            _,(hidden, _) = self.bilstm(uni_embeddings)
        
        sequence_output = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim = 1) # 得到整句话的表示
        
        if training and self.nn_drop_out > 0:
            sequence_output = self.dropout(sequence_output)
        logits = self.linear(sequence_output).reshape(-1)

        loss = nn.BCEWithLogitsLoss()(logits, input_tags.float())
        
        tag_scores = torch.sigmoid(logits)
        return tag_scores, loss
