import torch
import config
import logging
import numpy as np
from tqdm import tqdm
import torch.distributed as dist
from metric import get_micro, get_acc
from distributed_utils import is_main_process

# 打印完整的numpy array
np.set_printoptions(threshold=np.inf)

def epoch_train(train_loader, model, optimizer, scheduler, device, epoch):
    # set model to training mode
    model.train()
    train_losses = torch.zeros(1).to(device)
    train_size = torch.tensor(len(train_loader)).to(device)
    # 在主进程中打印训练进度
    if is_main_process():
        train_loader = tqdm(train_loader) 
    # train 
    for idx, batch in enumerate(train_loader):
        x, y = batch
        x.squeeze_()
        y.squeeze_()
        x.to(device)
        y.to(device) # 上GPU
        model.zero_grad()
        y_pred_scores, loss = model.forward(x, y)
        train_losses += loss.item()
        # 梯度反传
        loss.backward()
        # 优化更新
        optimizer.step()
        optimizer.zero_grad()
    # scheduler
    scheduler.step()
    # all reduce
    dist.all_reduce(train_losses, op=dist.ReduceOp.SUM)
    dist.all_reduce(train_size, op=dist.ReduceOp.SUM)
    train_loss = float(train_losses) / train_size     
    # 主进程打印输出
    
    if is_main_process():
        logging.info("Epoch: {}, train loss: {}".format(epoch, train_loss))
    


def train(train_loader, dev_loader, model, optimizer, scheduler, device):
    """train the model and test model performance"""
    best_val_f1 = 0.0
    patience_counter = 0
    # start training
    for epoch in range(1, config.epoch_num + 1):
        epoch_train(train_loader, model, optimizer, scheduler, device, epoch)
        with torch.no_grad():
            # dev loss calculation
            metric = dev(dev_loader, model, device)
            val_f1 = metric['f1']
            dev_loss = metric['loss']
            
            if is_main_process():
                logging.info("epoch: {}, f1 score: {}, dev loss: {}".format(epoch, val_f1, dev_loss))
            
            improve_f1 = val_f1 - best_val_f1
            if improve_f1 > 1e-5:
                best_val_f1 = val_f1
                if is_main_process():
                    torch.save(model, config.model_dir)
                    
                    logging.info("--------Save best model!--------")
                if improve_f1 < config.patience:
                    patience_counter += 1
                else:
                    patience_counter = 0
            else:
                patience_counter += 1
            # Early stopping 因为上面的评价指标都是统一在整个验证集上的，所以所有GPU上的进程会统一break
            if (patience_counter >= config.patience_num and epoch > config.min_epoch_num) or epoch == config.epoch_num:
                if is_main_process():
                    logging.info("Best val f1: {}".format(best_val_f1))
                break
    if is_main_process():
        logging.info("Training Finished!")


def dev(data_loader, model, device, mode = 'dev'):
    """test model performance on dev-set"""
    model.eval()
    true_tags = []
    pred_tags = []
    dev_losses = torch.zeros(1).to(device)   
    dev_size = torch.tensor(len(data_loader)).to(device)
    nb_correct = torch.zeros(1).to(device) 
    nb_pred = torch.zeros(1).to(device) 
    nb_true = torch.zeros(1).to(device)
    
    if is_main_process() or mode == 'test':
        data_loader = tqdm(data_loader) 
    for idx, batch in enumerate(data_loader):
        x, y = batch
        x.squeeze_()
        y.squeeze_()
        x.to(device)
        y.to(device) # 上GPU
        y_pred_scores, dev_loss = model.forward(x, y, training=False)
        dev_losses += dev_loss.item()
        y_pred_scores[y_pred_scores > 0.5] = 1
        y_pred_scores[y_pred_scores <= 0.5] = 0
        true_tags.extend(y)   # extend 带上中括号可以保持中括号里面是一个整体
        pred_tags.extend(y_pred_scores)
        # 计算损失
        dev_losses += dev_loss.item()
    assert len(pred_tags) == len(true_tags)
    
    nb_c, nb_p, nb_t = get_micro(true_tags, pred_tags)
    nb_correct += nb_c
    nb_pred += nb_p
    nb_true += nb_t
    
    if mode == "dev":
        # 统计整个开发集上的评价指标
        dist.all_reduce(dev_losses, op=dist.ReduceOp.SUM)
        dist.all_reduce(dev_size, op=dist.ReduceOp.SUM)
        dist.all_reduce(nb_correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(nb_pred, op=dist.ReduceOp.SUM)
        dist.all_reduce(nb_true, op=dist.ReduceOp.SUM)
        
    metrics = {}
    dev_loss = float(dev_losses) / dev_size
    p = nb_correct / nb_pred if nb_pred > 0 else 0  # 计算精确率
    r = nb_correct / nb_true if nb_true > 0 else 0  # 计算召回率
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0   # 计算f1
    metrics['p'], metrics['r'], metrics['f1'], metrics['loss'] = p.item(), r.item(), f1.item(), dev_loss.item()
    
    # 计算测试集上的正确率
    if mode == "test":
        metrics['acc'] = get_acc(true_tags, pred_tags).item()
    return metrics

    
