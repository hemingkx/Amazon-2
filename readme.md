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

| Exp  | max length | training set size | batch size | Training time | Accuracy |
| :--: | :--------: | :---------------: | :--------: | :-----------: | :------: |
|  1   |    196     |       60000       |     32     |  20min/epoch  |  0.8567  |
|  2   |            |                   |            |               |          |
|  3   |            |                   |            |               |          |

