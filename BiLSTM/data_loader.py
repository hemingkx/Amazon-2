import torch
import logging
from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer


class MyDataset(Dataset):
    def __init__(self, sentences, labels, vocab):
        self.vocab = vocab
        self.dataset = self.preprocess(sentences, labels)

    def preprocess(self, sentences, labels):
        """convert the data to ids"""
        dataset = []
        tokenizer = get_tokenizer('basic_english')
        for i in range(len(labels)):
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

    def collate_fn(self, batch):
        sequences = [x[0] for x in batch]
        labels = [x[1] for x in batch]
        batch_size = len(batch)

        # padding and mask
        length = max([len(s) for s in sequences])  # 该batch中最长的句子的单词数量
        word_ids = torch.LongTensor(batch_size, length).fill_(0)
        label_ids = torch.LongTensor(labels)
        for i, s in enumerate(sequences):
            word_ids[i, :len(s)] = torch.LongTensor(s)
        return [word_ids, label_ids]
