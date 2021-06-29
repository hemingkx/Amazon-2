import os
import config
import logging
import torch
import numpy as np

def get_micro(y_true, y_pred):
    
    y_true = np.array(y_true, dtype=np.int32)
    y_pred = np.array(y_pred, dtype=np.int32)
    nb_correct = np.sum(y_true & y_pred)  # 得到预测正确的数量
    nb_pred = np.sum(y_pred)  # 得到预测中认为正确的数量
    nb_true = np.sum(y_true)  # 得到标签中正确的数量
    return nb_correct, nb_pred, nb_true

def get_acc(y_true, y_pred):
    
    y_true = np.array(y_true, dtype=np.int32)
    y_pred = np.array(y_pred, dtype=np.int32)
    correct = np.sum(y_true == y_pred)  # 得到预测正确的数量
    acc = correct / len(y_true)
    return acc