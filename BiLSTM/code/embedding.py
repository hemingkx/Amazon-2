import config
import logging

import torch

import warnings
warnings.filterwarnings('ignore')

import numpy as np
from tqdm import tqdm

def embedding(vocab):
    vocab_size = len(vocab)
    embed_size = config.embedding_size
    weight = torch.randn(vocab_size, embed_size)
    cnt = 0
    char_set = set()
    
    pre_emb_file = open(config.embedding_dir)
    for char_emb in pre_emb_file:
        char_emb = char_emb.strip().split()
        char = char_emb[0]
        if vocab.word2id.__contains__(char) and char not in char_set:
            char_set.add(char)
            cnt += 1
            emb = np.array([float(num) for num in char_emb[1:]])
            index = vocab.word2id[char]
            weight[index, :] = torch.from_numpy(emb)    
    logging.info("--------Pretrained Embedding Loaded ! ({}/{})--------".format(cnt, len(vocab)))
    return weight # 返回预训练的字向量
