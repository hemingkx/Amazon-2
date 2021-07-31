import logging
import pandas as pd


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path) 
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def load_dataset(dataset_dir):
    dataset = pd.read_csv(dataset_dir, header=None)
    sentences = []
    labels = []
    for i in range(0, len(dataset)):
        sentences.append(str(dataset[1][i]) + '. ' + str(dataset[2][i]))   # 只要 content
        labels.append(dataset[0][i] - 1)   # 将 1 2 标签 转换成 0 1 标签
    logging.info("-------- Load Dataset! --------")
    return sentences, labels
