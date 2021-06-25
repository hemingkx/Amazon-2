# Amazon-2

This is our repository of Amazon-2 text classification task.

## Requirements

This repo was tested on Python 3.6+ and PyTorch 1.5.1. The main requirements are:

- tqdm
- scikit-learn
- pytorch >= 1.5.1
- 🤗transformers == 2.2.2

## Pretrained Model Required

需要提前下载BERT的预训练模型，包括

- pytorch_model.bin
- vocab.txt

放置在./pretrained_bert_models对应的预训练模型文件夹下。

## Experiments

| Exp  | max length | training set size | batch size | cls lr | Training time |        Accuracy         |
| :--: | :--------: | :---------------: | :--------: | :----: | :-----------: | :---------------------: |
|  1   |    196     |       60000       |     32     |  6e-4  |  20min/epoch  |         0.8567          |
|  2   |    256     |       60000       |     24     |  6e-4  |  30min/epoch  |         0.8580          |
|  3   |    196     |      360000       |     32     |  6e-4  |   2h/epoch    |         0.9080          |
|  4   |    196     |      3600000      |     32     |  6e-4  |   20h/epoch   | **0.9351(3 epoch now)** |

