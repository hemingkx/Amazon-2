import os
import config
import pandas
import logging
import numpy as np
from tqdm import tqdm


class Processor:
    def __init__(self, config):
        self.data_dir = config.data_dir
        self.config = config

    def process(self):
        """
        process train and test data
        """
        for file_name in self.config.files:
            self.preprocess(file_name)

    def preprocess(self, mode):
        """
        params:
            words：将csv文件每一行中的title和content合并，存储为words列表
            labels：标记文本对应的标签，存储为labels
        """
        input_dir = self.data_dir + str(mode) + '.csv'
        output_dir = self.data_dir + str(mode) + '.npz'
        if os.path.exists(output_dir) is True:
            return
        word_list = []
        label_list = []
        df = pandas.read_csv(input_dir, header=None)
        df.columns = ['label', 'title', 'content']
        for i in tqdm(range(0, int(len(df) / 60))):
            if pandas.isnull(df['title'][i]):
                words = list(df['content'][i])
            else:
                words = list(df['title'][i] + df['content'][i])
            if len(words) > self.config.max_len:
                # 最大长度截断
                words = words[:self.config.max_len]
            word_list.append(words)
            label_list.append(df['label'][i] - 1)
        # 保存成二进制文件
        np.savez_compressed(output_dir, words=word_list, labels=label_list)
        logging.info("--------{} data process DONE!--------".format(mode))

