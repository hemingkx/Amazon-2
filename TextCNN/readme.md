## Pretrained Model Required

需要提前下载预训练词向量：

- glove.6B.100d.txt

放置在 ./data 文件夹下

## Experiments

| Exp  | training set size | batch size | learning rate | training time | accuracy |  GPUs |
| ---- | ----------------- | ---------- | ------------- | ------------- | -------- | -------- |
| 1    | 60000         | 256      | 0.001      | 10s/epoch    | 0.885   |  2  |
| 2    | 360000        | 256      | 0.001      | 1min/epoch   | 0.911  |  3  |
| 3    | 3600000        | 256      | 0.001      | 10min/epoch   | 0.9337 |  3|
