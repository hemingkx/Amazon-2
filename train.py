import torch
import logging
import torch.nn as nn
from tqdm import tqdm

import config
import numpy as np
from model import BertForCls


def train_epoch(train_loader, model, optimizer, scheduler, epoch):
    # set model to training mode
    model.train()
    # step number in one epoch: 336
    train_losses = 0
    for idx, batch_samples in enumerate(tqdm(train_loader)):
        batch_data, batch_labels = batch_samples
        batch_masks = batch_data.gt(0)  # get padding mask
        # compute model output and loss
        loss = model(batch_data, token_type_ids=None, attention_mask=batch_masks, labels=batch_labels)[0]
        train_losses += loss.item()
        # clear previous gradients, compute gradients of all variables wrt loss
        model.zero_grad()
        loss.backward()
        # gradient clipping
        nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=config.clip_grad)
        # performs updates using calculated gradients
        optimizer.step()
        scheduler.step()
    train_loss = float(train_losses) / len(train_loader)
    logging.info("Epoch: {}, train loss: {}".format(epoch, train_loss))


def train(train_loader, dev_loader, model, optimizer, scheduler, model_dir):
    """train the model and test model performance"""
    # reload weights from restore_dir if specified
    if model_dir is not None and config.load_before:
        model = BertForCls.from_pretrained(model_dir)
        model.to(config.device)
        logging.info("--------Load model from {}--------".format(model_dir))
    best_val_acc = 0.0
    patience_counter = 0
    # start training
    for epoch in range(1, config.epoch_num + 1):
        train_epoch(train_loader, model, optimizer, scheduler, epoch)
        val_metrics = evaluate(dev_loader, model)
        val_acc = val_metrics['acc']
        logging.info("Epoch: {}, dev loss: {}, accuracy: {}".format(epoch, val_metrics['loss'], val_acc))
        improve_acc = val_acc - best_val_acc
        if improve_acc > 1e-5:
            best_val_acc = val_acc
            model.save_pretrained(model_dir)
            logging.info("--------Save best model!--------")
            if improve_acc < config.patience:
                patience_counter += 1
            else:
                patience_counter = 0
        else:
            patience_counter += 1
        # Early stopping and logging best f1
        if (patience_counter >= config.patience_num and epoch > config.min_epoch_num) or epoch == config.epoch_num:
            logging.info("Best val acc: {}".format(best_val_acc))
            break
    logging.info("Training Finished!")


def evaluate(dev_loader, model):
    # set model to evaluation mode
    model.eval()
    dev_losses = 0
    acc_num = 0
    total_num = 0

    with torch.no_grad():
        for idx, batch_samples in enumerate(dev_loader):
            batch_data, batch_labels = batch_samples
            batch_masks = batch_data.gt(0)  # get padding mask
            # compute model output and loss
            output = model(batch_data, token_type_ids=None, attention_mask=batch_masks, labels=batch_labels)
            loss = output[0]
            dev_losses += loss.item()
            # shape: (batch_size, num_labels)
            pred_probs = output[1].cpu().numpy()
            batch_preds = np.argmax(pred_probs, axis=1)
            batch_labels = batch_labels.cpu().numpy()
            # print("preds: ", batch_preds)
            # print("labels: ", batch_labels)
            acc_num += np.sum(batch_preds == batch_labels)
            total_num += len(batch_labels)
    metrics = {'loss': float(dev_losses) / len(dev_loader)}
    acc = acc_num / total_num
    metrics['acc'] = acc
    return metrics
