import logging
import numpy as np
from collections import Counter
from torchtext.data.utils import get_tokenizer


class Vocabulary:
    """构建词表"""
    def __init__(self, config):
        self.config = config
        self.vocab_path = config.vocab_path
        self.word2id = {}
        self.id2word = {}

    def __len__(self):
        return len(self.word2id)

    def vocab_size(self):
        return len(self.word2id)

    def get_vocab(self, sentences_train, sentences_test):
        """
        进一步处理，将word和label转化为id
        word2id: dict,每个字对应的序号
        id2word: dict,每个序号对应的字
        保存为二进制文件
        """

        # 如果没有处理好的二进制文件，就处理原始的npz文件
        tokenizer = get_tokenizer('basic_english')
        word_counter = Counter()
        for sentence in sentences_train:
            word_list = tokenizer(sentence)
            word_counter.update(word_list)
        for sentence in sentences_test:
            word_list = tokenizer(sentence)
            word_counter.update(word_list)
        # 通过字的频率倒序排列，并转化为列表，类似于：[('i', 2),('good', 2),('you', 2),('am', 1)]
        sorted_word_counter = sorted(word_counter.items(), key=lambda e: e[1], reverse=True)
        # 构建word2id字典
        index = 0
        self.word2id["<pad>"] = index
        index += 1
        for elem in sorted_word_counter:
            self.word2id[elem[0]] = index
            index += 1
        # 构建id2word字典
        self.id2word = {idx: word for word, idx in list(self.word2id.items())}
        # 保存为二进制文件
        np.savez_compressed(self.vocab_path, word2id=self.word2id, id2word=self.id2word)
        logging.info("-------- Vocabulary Build! --------")
