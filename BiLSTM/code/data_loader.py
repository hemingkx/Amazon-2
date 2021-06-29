import torch
import logging
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_padded_sequence
from torchtext.data.utils import get_tokenizer

class EEDataset(Dataset):
    def __init__(self, sentences, labels, vocab):
        self.size = len(labels)
        self.vocab = vocab
        self.dataset = self.preprocess(sentences, labels) # 序列化数据和标签

    def preprocess(self, sentences, labels):
        """convert the data to ids"""
        dataset = []
        tokenizer = get_tokenizer('basic_english')
        for i in range(self.size):
            sentence = sentences[i]
            word_list = tokenizer(sentence)
            sequence = [self.vocab.word2id[word] for word in word_list]
            dataset.append((sequence, labels[i]))
        logging.info("-------- Process Done! --------")
        return dataset

    def __getitem__(self, idx):
        sentence = self.dataset[idx][0]
        label = self.dataset[idx][1]
        return [sentence, label]

    def __len__(self):
        return len(self.dataset)

    def get_long_tensor(self, sequences, labels, batch_size, lens):
        # padding and mask
        length = max(lens) # 该batch中最长的句子的单词数量
        sequences_pad = torch.LongTensor(batch_size, length).fill_(0)
        
        labels = torch.LongTensor(labels)

        for i, s in enumerate(sequences):
            sequences_pad[i, :len(s)] = torch.LongTensor(s)
        
        # 下面将数据根据序列长度，从长到短排列
        # 方便变长RNN模型使用数据
        data = pack_padded_sequence(sequences_pad, lens, batch_first=True, enforce_sorted=False)
        word_tokens = sequences_pad[data.sorted_indices]
        label_tokens = labels[data.sorted_indices]
        sorted_lens = torch.LongTensor(lens)[data.sorted_indices]
        
        return word_tokens, label_tokens, sorted_lens
        

    def collate_fn(self, batch):
        # 自定义 collate_fn
        sequences = [x[0] for x in batch]
        labels = [x[1] for x in batch]
        lens = [len(s) for s in sequences]
        batch_size = len(batch)

        word_ids, label_ids, lens = self.get_long_tensor(sequences, labels, batch_size, lens)

        return [word_ids, label_ids, lens]