import torch
import numpy as np
from transformers import BertTokenizer
from torch.utils.data import Dataset


class AmazonDataset(Dataset):
    def __init__(self, words, labels, config, word_pad_idx=0):
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_model, do_lower_case=True)
        self.dataset = self.preprocess(words, labels)
        self.word_pad_idx = word_pad_idx
        self.device = config.device

    def preprocess(self, origin_sentences, labels):
        """
        Maps tokens to their indices and stores them in the dict data.
        """
        data = []
        sentences = []
        for line in origin_sentences:
            # replace each token by its index
            words = []
            word_lens = []
            for token in line:
                words.append(self.tokenizer.tokenize(token))
                word_lens.append(len(token))
            # 变成单个字的列表，开头加上[CLS]
            words = ['[CLS]'] + [item for token in words for item in token]
            sentences.append(self.tokenizer.convert_tokens_to_ids(words))
        for sentence, label in zip(sentences, labels):
            data.append((sentence, label))
        return data

    def __getitem__(self, idx):
        """sample data to get batch"""
        word = self.dataset[idx][0]
        label = self.dataset[idx][1]
        return [word, label]

    def __len__(self):
        """get dataset size"""
        return len(self.dataset)

    def collate_fn(self, batch):
        """
        process batch data, including:
            1. padding: 将每个batch的data padding到同一长度（batch中最长的data长度）
            2. tensor：转化为tensor
        """
        sentences = [x[0] for x in batch]
        batch_labels = [x[1] for x in batch]

        # batch length
        batch_len = len(sentences)
        # compute length of longest sentence in batch
        max_len = max([len(s) for s in sentences])
        # padding data 初始化
        batch_data = self.word_pad_idx * np.ones((batch_len, max_len))
        # padding
        for j in range(batch_len):
            cur_len = len(sentences[j])
            batch_data[j][:cur_len] = sentences[j]

        # convert data to torch LongTensors
        batch_data = torch.tensor(batch_data, dtype=torch.long)
        batch_labels = torch.tensor(batch_labels, dtype=torch.long)

        # shift tensors to GPU if available
        batch_data = batch_data.to(self.device)
        batch_labels = batch_labels.to(self.device)
        return [batch_data, batch_labels]
