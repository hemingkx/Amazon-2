import torch
import config
import logging
from sklearn.metrics import accuracy_score

from tqdm import tqdm
import numpy as np

# 打印完整的numpy array
np.set_printoptions(threshold=np.inf)


def epoch_train(train_loader, model, optimizer, scheduler, device, epoch):
    # set model to training mode
    model.train()
    train_loss = 0.0
    for idx, batch_samples in enumerate(tqdm(train_loader)):
        x, y = batch_samples
        x = x.to(device)
        y = y.to(device)
        model.zero_grad()
        _, loss = model.forward(x, y)
        train_loss += loss.item()
        # 梯度反传
        loss.backward()
        # 优化更新
        optimizer.step()
        optimizer.zero_grad()
    # scheduler
    scheduler.step()
    train_loss = float(train_loss) / len(train_loader)

    logging.info("epoch: {}, train loss: {}".format(epoch, train_loss))


def train(train_loader, dev_loader, model, optimizer, scheduler, device):
    """train the model and test model performance"""
    best_accuracy = 0.0
    patience_counter = 0
    # start training
    for epoch in range(1, config.epoch_num + 1):
        epoch_train(train_loader, model, optimizer, scheduler, device, epoch)
        with torch.no_grad():
            # dev loss calculation
            dev_accuracy, dev_loss = val(dev_loader, model, device)

            logging.info("epoch: {}, dev accuracy: {}, dev loss: {}".format(epoch, dev_accuracy, dev_loss))

            improve_accuracy = dev_accuracy - best_accuracy
            if improve_accuracy > 1e-5:
                best_accuracy = dev_accuracy

                torch.save(model, config.model_dir)

                logging.info("--------Save best model!--------")
                if improve_accuracy < config.patience:
                    patience_counter += 1
                else:
                    patience_counter = 0
            else:
                patience_counter += 1
            # Early stopping and logging best f1
            if (patience_counter >= config.patience_num and epoch > config.min_epoch_num) or epoch == config.epoch_num:
                logging.info("Best val f1: {}".format(best_accuracy))
                break
    logging.info("Training Finished!")


def val(data_loader, model, device):
    """test model performance on dev-set"""
    model.eval()
    true_tags = []
    pred_tags = []
    dev_losses = 0
    for idx, batch_samples in enumerate(tqdm(data_loader)):
        words, labels = batch_samples
        words = words.to(device)
        labels = labels.to(device)
        y_pred_scores, dev_loss = model.forward(words, labels, training=False)
        y_pred_scores[y_pred_scores > 0.5] = 1
        y_pred_scores[y_pred_scores <= 0.5] = 0
        targets = y_pred_scores
        true_tags.extend(labels.cpu().numpy())
        pred_tags.extend(targets.cpu().numpy())
        # 计算损失
        dev_losses += dev_loss.item()
    assert len(pred_tags) == len(true_tags)
    accuracy = accuracy_score(true_tags, pred_tags)
    loss = float(dev_losses) / len(data_loader)

    return accuracy, loss


def load_model(model_dir, device):
    # Prepare model
    model = torch.load(model_dir)
    model.to(device)
    logging.info("--------Load model from {}--------".format(model_dir))
    return model
