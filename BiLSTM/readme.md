## Pretrained Model Required

需要提前下载预训练词向量：

- glove.6B.50d.txt

放置在 ./embedding/glove.6B 文件夹下

## Experiments

| Exp  | training set size | batch size | learning rate | training time | accuracy |
| ---- | ----------------- | ---------- | ------------- | ------------- | -------- |
| 1    | 60000             | 2048       | 0.001         | 16s/epoch     | 0.8385   |
| 2    | 360000            | 2048       | 0.001         | 1.5min/epoch  | 0.9043   |
| 3    | 3600000           | 2048       | 0.001         | 19min/epoch   | 0.9362   |
