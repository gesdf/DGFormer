import os
import argparse
from time import time
from datetime import datetime
from collections import defaultdict
from progressbar import ProgressBar

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from timm.scheduler import CosineLRScheduler, PlateauLRScheduler

from models import build_model
from dataset import  create_dataloader
import utils
from utils import TensorboardManager, RecordExp, flatten_dict, merge_dicts
from loss import BinaryCEWithLogitLoss
from optimizer import build_optimizer
from configs import get_cfg_defaults
import matplotlib.pyplot as plt
import seaborn as sns

DEVICE = ""


def get_inp(data_batch, device):
    predi = data_batch['predicate']['idx'].to(device, non_blocking=True)
    subj_bbox = data_batch['subject']['bbox'].to(device, non_blocking=True)
    obj_bbox = data_batch['object']['bbox'].to(device, non_blocking=True)
    rgb = data_batch['img'].to(device, non_blocking=True)
    subl=data_batch['subject']['idx'].to(device, non_blocking=True)
    objl=data_batch['object']['idx'].to(device, non_blocking=True)
    t_s=data_batch['subject']['t'].to(device, non_blocking=True)
    t_o=data_batch['object']['t'].to(device, non_blocking=True)
    union_bbox = data_batch['predicate']['bbox'].to(device, non_blocking=True)

    inp = {
        "full_im": rgb,
        "bbox_s": subj_bbox,
        "bbox_o": obj_bbox,
        "predicate": predi,
        "subject_label":subl,
        "object_label":objl,
        "subject_t":t_s,
        "object_t":t_o,
        "union_bbox": union_bbox
    }
    # Add depth data if available
    if 'depth' in data_batch:
        depth = data_batch['depth'].to(device, non_blocking=True)
        inp["full_depth"] = depth
    
    return inp

# 添加一个新函数：冻结模型所有参数
def freeze_all_parameters(model):
    """冻结模型的所有参数"""
    for param in model.parameters():
        param.requires_grad = False
    print("已冻结模型所有参数")

# 新增：参数冻结与解冻辅助函数
def freeze_parameters(model, param_names):
    """冻结指定参数"""
    for name, param in model.named_parameters():
        if any(p_name in name for p_name in param_names):
            param.requires_grad = False
            print(f"已冻结参数: {name}")


def unfreeze_parameters(model, param_names):
    """解冻指定参数"""
    for name, param in model.named_parameters():
        if any(p_name in name for p_name in param_names):
            param.requires_grad = True
            print(f"已解冻参数: {name}")


def freeze_predicate_specific_parameters(model):
    """冻结所有谓词特化参数"""
    # freeze_parameters(model, ['predicate_cls_tokens', 'readout_heads'])
    freeze_parameters(model, ['predicate_cls_tokens'])
    print("已冻结predicate_cls_tokens")


def unfreeze_predicate_specific_parameters(model):
    """解冻所有谓词特化参数"""
    unfreeze_parameters(model, ['predicate_cls_tokens', 'readout_heads'])
    # unfreeze_parameters(model, ['predicate_cls_tokens'])
    print("已解冻谓词特化参数(cls_tokens和分类头)")


def freeze_shared_parameters(model):
    """冻结共享参数"""
    freeze_parameters(model, ['shared_cls_token'])
    print("已冻结shared_cls_token")


def unfreeze_shared_parameters(model):
    """解冻共享参数"""
    unfreeze_parameters(model, ['shared_cls_token'])
    print("已解冻shared_cls_token")


def freeze_specific_predicate_head(model, predicate_idx):
    """冻结除了指定谓词外的所有谓词分类头"""
    for i, head in enumerate(model.readout_heads):
        if i != predicate_idx:
            for param in head.parameters():
                param.requires_grad = False
    print(f"已冻结除谓词 {predicate_idx} 外的所有分类头")

def plot_channel_attention(attn_weights, num_channels):
    """
    绘制通道间的注意力热图
    Args:
        attn_weights: 注意力权重 [num_channels, num_channels]
        num_channels: 通道数
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(attn_weights.cpu().detach().numpy(), annot=False, cmap="viridis")
    plt.title("Channel Attention Heatmap")
    plt.xlabel("Channels")
    plt.ylabel("Channels")
    plt.show()

def plot_spatial_attention(attn_weights, feature_map):
    """
    绘制空间注意力热图
    Args:
        attn_weights: 通道注意力权重 [num_channels]
        feature_map: 特征图 [num_channels, H, W]
    """
    weighted_map = torch.einsum('c,chw->hw', attn_weights, feature_map)

    weighted_map = (weighted_map - weighted_map.min()) / (weighted_map.max() - weighted_map.min())

    plt.imshow(weighted_map.cpu().detach().numpy(), cmap="viridis")
    plt.title("Spatial Attention Heatmap")
    plt.colorbar()
    plt.show()

def extract_attention_weights(model, layer_idx):
    """
    从模型中提取指定层的注意力权重
    Args:
        model: Transformer 模型
        layer_idx: 要提取的层索引
    Returns:
        attn_weights: 注意力权重 [num_heads, num_tokens, num_tokens]
    """
    attn_weights = model.blocks[layer_idx].attn.attn_weights 
    return attn_weights.mean(dim=1)  

def visualize_attention(model, feature_map, layer_idx=0):
    """
    可视化通道间注意力和空间注意力
    Args:
        model: Transformer 模型
        feature_map: 特征图 [num_channels, H, W]
        layer_idx: 要提取的注意力层索引
    """
    attn_weights = extract_attention_weights(model, layer_idx)

    num_channels = feature_map.shape[0]

    plot_channel_attention(attn_weights[0], num_channels)

    plot_spatial_attention(attn_weights[0], feature_map)

def validate(loader, model, criterion, device):
    """
    :param loader:
    :param model:
    :param criterion:
    :param device:
    :return:
    """
    model.eval()
    correct = []
    tp = []
    fp = []
    p = []
    losses = []
    # dictionary storing correct list relation wise
    correct_rel = defaultdict(list)

    with torch.no_grad():
        _bar = ProgressBar(max_value=len(loader))
        for i, data_batch in enumerate(loader):
            label = data_batch['label'].to(device, non_blocking=True)
            inp = get_inp(data_batch, device)

            output = model(**inp)
            loss = criterion(output, label.to(dtype=torch.float32), reduction='none')
            losses.append(loss)

            logit = output[0] if isinstance(output, tuple) else output
            batch_correct = (((logit > 0) & (label == True))
                             | ((logit <= 0) & (label == False))).tolist()
            tp.extend(((logit > 0) & (label == True)).tolist())
            fp.extend(((logit > 0) & (label == False)).tolist())

            correct.extend(batch_correct)
            p.extend((label == True).tolist())
            for pred_name, _correct in zip(data_batch['predicate']['name'],
                                           batch_correct):
                correct_rel[pred_name].append(_correct)
            _bar.update(i)

    acc = sum(correct) / len(correct)
    pre = sum(tp) / (sum(tp) + sum(fp) + 0.00001)
    rec = sum(tp) / sum(p)
    f1 = (2 * pre * rec) / (pre + rec + 0.00001)
    acc_rel = {x: sum(y)/len(y) for x, y in correct_rel.items()}
    acc_rel_avg = sum(acc_rel.values()) / len(acc_rel.values())
    losses = torch.cat(losses, dim=0)

    return acc, pre, rec, f1, acc_rel, acc_rel_avg, losses


def train(loader, model, criterion, optimizer, device):
    model.train()
    time_forward = 0
    time_backward = 0
    time_data_loading = 0
    losses = []
    avg_loss = []
    correct = []
    tp = []
    fp = []
    p = []
    # dictionary storing correct list relation wise
    correct_rel = defaultdict(list)

    time_last_batch_end = time()
    for i, data_batch in enumerate(loader):
        time_start = time()
        label = data_batch['label'].to(device, non_blocking=True)
        inp = get_inp(data_batch, device)
        output = model(**inp)

        loss = criterion(output, label.to(dtype=torch.float32))
        time_forward += (time() - time_start)

        logit = output[0] if isinstance(output, tuple) else output
        avg_loss.append(loss.item())
        losses.append(loss.item())
        batch_correct = (((logit > 0) & (label == True))
                         | ((logit <= 0) & (label == False))).tolist()
        correct.extend(batch_correct)
        tp.extend(((logit > 0) & (label == True)).tolist())
        p.extend((label == True).tolist())
        fp.extend(((logit > 0) & (label == False)).tolist())
        for pred_name, _correct in zip(data_batch['predicate']['name'],
                                       batch_correct):
            correct_rel[pred_name].append(_correct)

        optimizer.zero_grad()
        time_start = time()
        loss.backward()
        # 添加梯度裁剪以提高训练稳定性
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        time_backward += (time() - time_start)
        optimizer.step()
        time_data_loading += (time_start - time_last_batch_end)
        time_last_batch_end = time()

        if i % 50 == 0:
            print(
                '[%d/%d] Loss = %.02f, Forward time = %.02f, Backward time = %.02f, Data loading time = %.02f' \
                % (i, len(loader), np.mean(avg_loss), time_forward,
                   time_backward, time_data_loading))

            avg_loss = []

    acc = sum(correct) / len(correct)
    pre = sum(tp) / (sum(tp) + sum(fp) + 0.00001)
    rec = sum(tp) / sum(p)
    f1 = (2 * pre * rec) / (pre + rec + 0.00001)
    acc_rel = {x: sum(y)/len(y) for x, y in correct_rel.items()}
    acc_rel_avg = sum(acc_rel.values()) / len(acc_rel.values())
    losses = sum(losses) / len(losses)

    return acc, pre, rec, f1, acc_rel, acc_rel_avg, losses


def train_by_predicate(loader, model, criterion, optimizer, device, predicate_name, max_batches=None):
    """
    针对特定谓词的数据进行专门训练
    
    Args:
        loader: 特定谓词的数据加载器
        model: 模型
        criterion: 损失函数
        optimizer: 优化器
        device: 设备
        predicate_name: 谓词名称
        max_batches: 最大训练批次数，None表示训练全部
    
    Returns:
        acc: 准确率
        loss_avg: 平均损失
    """
    model.train()
    losses = []
    correct = []
    tp = []
    fp = []
    p = []
    
    print(f"\n开始训练谓词 '{predicate_name}' ({len(loader.dataset)} 个样本)...")
    
    # 训练进度条
    _bar = ProgressBar(max_value=min(len(loader), max_batches) if max_batches else len(loader))
    
    for i, data_batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
            
        # 加载数据到设备
        label = data_batch['label'].to(device, non_blocking=True)
        inp = get_inp(data_batch, device)
        
        # 前向传播
        output = model(**inp)
        loss = criterion(output, label.to(dtype=torch.float32))
        
        # 计算准确率
        logit = output[0] if isinstance(output, tuple) else output
        batch_correct = (((logit > 0) & (label == True)) | 
                        ((logit <= 0) & (label == False))).tolist()
        correct.extend(batch_correct)
        tp.extend(((logit > 0) & (label == True)).tolist())
        p.extend((label == True).tolist())
        fp.extend(((logit > 0) & (label == False)).tolist())
        
        # 记录损失
        losses.append(loss.item())
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪以提高稳定性
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        _bar.update(i)
        
        # # 每10批次打印一次
        # if i % 10 == 0:
        #     print(f"  [{i}/{len(loader)}] {predicate_name} Loss = {loss.item():.4f}")
    
    # 计算指标
    acc = sum(correct) / len(correct) if correct else 0
    pre = sum(tp) / (sum(tp) + sum(fp) + 0.00001) if tp else 0
    rec = sum(tp) / sum(p) if p else 0
    f1 = (2 * pre * rec) / (pre + rec + 0.00001) if pre + rec > 0 else 0
    loss_avg = sum(losses) / len(losses) if losses else 0
    
    print(f"谓词 '{predicate_name}' 训练完成: Acc = {acc:.4f}, Loss = {loss_avg:.4f}, F1 = {f1:.4f}")
    return acc, loss_avg, f1


def validate_by_predicate(loader, model, criterion, device, predicate_name):
    """
    针对特定谓词的数据进行验证
    """
    model.eval()
    correct = []
    losses = []
    tp = []
    fp = []
    p = []
    
    print(f"\n验证谓词 '{predicate_name}' ({len(loader.dataset)} 个样本)...")
    
    with torch.no_grad():
        for i, data_batch in enumerate(loader):
            # 加载数据到设备
            label = data_batch['label'].to(device, non_blocking=True)
            inp = get_inp(data_batch, device)
            
            # 前向传播
            output = model(**inp)
            loss = criterion(output, label.to(dtype=torch.float32), reduction='none')
            losses.append(loss)
            
            # 计算准确率
            logit = output[0] if isinstance(output, tuple) else output
            batch_correct = (((logit > 0) & (label == True)) | 
                            ((logit <= 0) & (label == False))).tolist()
            correct.extend(batch_correct)
            tp.extend(((logit > 0) & (label == True)).tolist())
            fp.extend(((logit > 0) & (label == False)).tolist())
            p.extend((label == True).tolist())
    
    # 计算指标
    acc = sum(correct) / len(correct) if correct else 0
    pre = sum(tp) / (sum(tp) + sum(fp) + 0.00001) if tp else 0
    rec = sum(tp) / sum(p) if p else 0
    f1 = (2 * pre * rec) / (pre + rec + 0.00001) if pre + rec > 0 else 0
    losses = torch.cat(losses, dim=0) if losses else torch.tensor([0.0])
    loss_avg = losses.mean().item()
    
    print(f"谓词 '{predicate_name}' 验证结果: Acc = {acc:.4f}, Loss = {loss_avg:.4f}, F1 = {f1:.4f}")
    return acc, loss_avg, f1


def save_checkpoint(epoch, model, optimizer, acc, cfg):
    path = f"{cfg.EXP.OUTPUT_DIR}/{cfg.EXP.EXP_ID}/model_best.pth"
    torch.save({
        'cfg': vars(cfg),
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'accuracy': acc,
    }, path)
    print('Checkpoint saved to %s' % path)


def load_best_checkpoint(model, cfg, device):
    path = f"{cfg.EXP.OUTPUT_DIR}/{cfg.EXP.EXP_ID}/model_best.pth"
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state'])
    print('Checkpoint loaded from %s' % path)
    model.to(device)


def load_checkpoint(model, model_path, device):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state'])
    print('Checkpoint loaded from %s' % model_path)
    model.to(device)


def entry_train(cfg, device, record_file=""):
    train_valid = cfg.DATALOADER.get("train_valid", False)
    skip_normal_training = cfg.TRAIN.get("skip_normal_training", False)
    predicate_specific_training = cfg.TRAIN.get("predicate_specific_training", False)
    predicate_specific_start_epoch = cfg.TRAIN.get("predicate_specific_start_epoch", 3)
    predicate_specific_frequency = cfg.TRAIN.get("predicate_specific_frequency", 1)
    predicate_specific_batches = cfg.TRAIN.get("predicate_specific_batches", 50)
    


     
    predicate_specific_started = False
    loader_train = create_dataloader(split='train', **cfg.DATALOADER)
    dataset_train = loader_train.dataset
    if predicate_specific_training:
        predicate_groups = dataset_train.group_by_predicate()
        print("\n按谓词训练已启用. 各谓词样本数量:")
        for pred_name, indices in predicate_groups.items():
            print(f"  • '{pred_name}': {len(indices)} 个样本")
    if train_valid:
        loader_valid = create_dataloader('test', **cfg.DATALOADER)
    else:
        loader_valid = create_dataloader('valid', **cfg.DATALOADER)
    
    loader_test = create_dataloader('test', **cfg.DATALOADER)

    model = build_model(cfg)
    model.to(device)

    if utils.is_dist_avail_and_initialized():
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    optimizer = build_optimizer(cfg, model)
    if cfg.TRAIN.scheduler == 'plateau':
        scheduler = PlateauLRScheduler(optimizer, mode='max', decay_rate=cfg.TRAIN.lr_decay_ratio, patience_t=cfg.TRAIN.patience, warmup_t=cfg.TRAIN.warmup, verbose=True)
    elif cfg.TRAIN.scheduler == 'cosine':
        scheduler = CosineLRScheduler(optimizer, t_initial=cfg.TRAIN.num_epochs, warmup_t=cfg.TRAIN.warmup, lr_min=1e-7)
    else:
        raise ValueError("不支持的学习率调度器类型")

    if utils.is_dist_avail_and_initialized():
        model = DDP(model, device_ids=[utils.get_rank()], output_device=utils.get_rank(), find_unused_parameters=True)
        model_without_ddp = model.module
    else:
        model_without_ddp = model
    print(model)

    criterion = BinaryCEWithLogitLoss()

    best_acc_rel_avg_valid = -1
    best_epoch_rel_avg_valid = 0
    best_acc_rel_avg_test = -1
    early_stop_flag = torch.zeros(1).to(device)

    # 设置TensorBoard记录
    log_dir = f"{cfg.EXP.OUTPUT_DIR}/{cfg.EXP.EXP_ID}"
    os.makedirs(log_dir, exist_ok=True)
    tb = TensorboardManager(log_dir)
    with open(os.path.join(log_dir, 'config.yaml'), 'w') as f:
        cfg.dump(stream=f)
        
    # 训练循环
    for epoch in range(cfg.TRAIN.num_epochs):
        print('\n' + '='*50)
        print(f'Epoch #{epoch}')
        print('='*50)
        
        if hasattr(loader_train.sampler, 'set_epoch'):
            loader_train.sampler.set_epoch(epoch)

        is_predicate_specific_epoch = predicate_specific_training and epoch >= predicate_specific_start_epoch and epoch % predicate_specific_frequency == 0

        if is_predicate_specific_epoch:
            predicate_specific_started = True
        if not skip_normal_training and not predicate_specific_started:
            print('正常训练阶段..')
            freeze_predicate_specific_parameters(model_without_ddp)

            unfreeze_shared_parameters(model_without_ddp)
            trainable_params = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
            print(f"正常训练阶段可训练参数数量: {trainable_params}")
            
            (acc_train, pre_train, rec_train, f1_train, acc_rel_train,
            acc_rel_avg_train, loss_train) = train(loader_train, model, criterion, optimizer, device)
            print(f'Train, acc avg: {acc_rel_avg_train} acc: {acc_train},'
                f' pre: {pre_train}, rec: {rec_train}, f1: {f1_train}, loss: {loss_train}, lr: {optimizer.param_groups[-1]["lr"]}')
            print({x: round(y, 3) for x, y in acc_rel_train.items()})
            tb.update('train', epoch, {'acc': acc_train, 'lr': optimizer.param_groups[-1]['lr']})
        elif predicate_specific_started and not is_predicate_specific_epoch:
            print('已开始谓词特化训练，跳过常规训练阶段...')
            # 设置默认值，避免后面引用未定义变量
            acc_train = 0.0
            acc_rel_avg_train = 0.0
        else:
            print('跳过正常训练阶段，直接进行按谓词训练...')
            # 设置默认值，避免后面引用未定义变量
            acc_train = 0.0
            acc_rel_avg_train = 0.0
            
        # 按谓词特化训练阶段
        if is_predicate_specific_epoch:
            print('\n' + '-'*50)
            print('按谓词特化训练阶段')
            print('-'*50)

            freeze_all_parameters(model_without_ddp)
            unfreeze_predicate_specific_parameters(model_without_ddp)

            trainable_params = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
            print(f"谓词特化训练阶段可训练参数数量: {trainable_params}")
            predicate_metrics = {}
            
            # 对每个谓词单独训练
            for pred_idx, pred_name in enumerate(dataset_train.predicates):
                # 创建特定谓词的数据加载器
                pred_loader = dataset_train.get_grouped_loader(
                    pred_name, 
                    batch_size=cfg.DATALOADER.batch_size,
                    num_workers=cfg.DATALOADER.num_workers
                )
                
                if len(pred_loader.dataset) < cfg.DATALOADER.batch_size:
                    adjusted_batch_size = max(4, len(pred_loader.dataset) // 2)
                    print(f"谓词 '{pred_name}' 样本数量不足，使用较小批次大小: {adjusted_batch_size}")
                    pred_loader = dataset_train.get_grouped_loader(
                        pred_name, 
                        batch_size=adjusted_batch_size,
                        num_workers=cfg.DATALOADER.num_workers
                    )
                
                pred_acc, pred_loss, pred_f1 = train_by_predicate(
                    pred_loader, model_without_ddp if utils.is_dist_avail_and_initialized() else model, 
                    criterion, optimizer, device, pred_name, 
                    max_batches=predicate_specific_batches
                )
                predicate_metrics[pred_name] = {
                    'acc': pred_acc,
                    'loss': pred_loss,
                    'f1': pred_f1
                }

                tb.update(f'train_pred_{pred_name}', epoch, {
                    'acc': pred_acc, 
                    'loss': pred_loss,
                    'f1': pred_f1
                })
            
            print('\n按谓词训练结果摘要:')
            for pred_name, metrics in predicate_metrics.items():
                print(f"  • {pred_name}: Acc = {metrics['acc']:.4f}, F1 = {metrics['f1']:.4f}")
            unfreeze_parameters(model_without_ddp, ['shared_cls_token', 'predicate_cls_tokens', 'readout_heads'])

        # 验证阶段
        acc_valid_tensor = torch.zeros(1).to(device)
        if utils.get_rank() == 0:
            print('\n验证阶段..')
            (acc_valid, pre_valid, rec_valid, f1_valid, acc_rel_valid,
             acc_rel_avg_valid, loss_valid) = validate(loader_valid, model_without_ddp, criterion, device)
            print(f'Valid, acc avg: {acc_rel_avg_valid} acc: {acc_valid},'
                  f' pre: {pre_valid}, rec: {rec_valid}, f1: {f1_valid}, loss: {loss_valid.mean().item()}')
            print({x: round(y, 3) for x, y in acc_rel_valid.items()})
            tb.update('val', epoch, {'acc': acc_valid})
            acc_valid_tensor = torch.tensor([acc_valid]).to(device)

            # 测试阶段
            print('\n测试阶段..')
            (acc_test, pre_test, rec_test, f1_test, acc_rel_test,
             acc_rel_avg_test, loss_test) = validate(loader_test, model_without_ddp, criterion, device)
            print(f'Test, acc avg: {acc_rel_avg_test} acc: {acc_test},'
                  f' pre: {pre_test}, rec: {rec_test}, f1: {f1_test}, loss: {loss_test.mean().item()}')
            print({x: round(y, 3) for x, y in acc_rel_test.items()})

            # 保存最佳模型
            if acc_rel_avg_valid > best_acc_rel_avg_valid:
                print('准确率已提高，保存模型')
                best_acc_rel_avg_valid = acc_rel_avg_valid
                best_epoch_rel_avg_valid = epoch

                save_checkpoint(epoch, model_without_ddp, optimizer, acc_rel_avg_valid, cfg)
            if acc_rel_avg_test > best_acc_rel_avg_test:
                best_acc_rel_avg_test = acc_rel_avg_test

            # 早停判断
            if (epoch - best_epoch_rel_avg_valid) > cfg.TRAIN.early_stop:
                early_stop_flag += 1

        # 同步分布式训练
        utils.synchronize()
        if utils.is_dist_avail_and_initialized():
            torch.distributed.all_reduce(acc_valid_tensor, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(early_stop_flag, op=torch.distributed.ReduceOp.SUM)
            if early_stop_flag >= 1:
                print(f"提前停止在第 {epoch} 个epoch，因为验证准确率在 {cfg.TRAIN.early_stop} 个epoch内没有提高。")
                break
        else:
            if early_stop_flag == 1:
                print(f"提前停止在第 {epoch} 个epoch，因为验证准确率在 {cfg.TRAIN.early_stop} 个epoch内没有提高。")
                break

        # 更新学习率
        scheduler.step(epoch=epoch+1, metric=acc_valid_tensor)

    # 训练结束，使用最佳模型进行最终测试
    if utils.get_rank() == 0:
        print('\n最终测试阶段..')
        load_best_checkpoint(model_without_ddp, cfg, device)
        (acc_test, pre_test, rec_test, f1_test, acc_rel_test,
         acc_rel_avg_test, loss_test) = validate(loader_test, model_without_ddp, criterion, device)
        print(f'最佳验证准确率: {best_acc_rel_avg_valid}')
        print(f'最佳测试准确率: {best_acc_rel_avg_test}')
        print(f'最佳模型测试结果: acc avg: {acc_rel_avg_test}, acc: {acc_test},'
              f' pre: {pre_test}, rec: {rec_test}, f1: {f1_test}, loss_test: {loss_test.mean().item()}')
        print({x: round(y, 3) for x, y in acc_rel_test.items()})

        # 记录实验结果
        if record_file != "":
            exp = RecordExp(record_file)
            exp.record_param(flatten_dict(dict(cfg)))
            exp.record_result({
                "final_train": acc_rel_avg_train,
                "best_val": best_acc_rel_avg_valid,
                "best_test": best_acc_rel_avg_test,
                "final_test": acc_rel_avg_test
            })


def entry_test(cfg, model_path, device):
    loader_test, _, _ = create_dataloader('test', **cfg.DATALOADER)
    criterion = BinaryCEWithLogitLoss()
    model = build_model(cfg)
    model.to(device)
    load_checkpoint(model, model_path, device)

    print('\nTesting..')
    (acc_test, pre_test, rec_test, f1_test, acc_rel_test,
     acc_rel_avg_test, loss_test) = validate(loader_test, model, criterion, device)
    print(f'Test at best valid, acc avg: {acc_rel_avg_test}, acc: {acc_test}, '
          f'pre: {pre_test}, rec: {rec_test}, f1: {f1_test}, loss: {loss_test.mean().item()}')
    print({x: round(y, 3) for x, y in acc_rel_test.items()})
    for x, y in acc_rel_test.items():
        print(x, ':', round(y, 3))
      # 可视化注意力
    print("\nVisualizing Attention...")
    # 从测试集中获取一个样本
    data_batch = next(iter(loader_test))
    inp = get_inp(data_batch, device)
    feature_map = inp["full_im"][0]  # 假设 full_im 是 [B, C, H, W]，取第一个样本
    visualize_attention(model, feature_map, layer_idx=0)

    return {'acc avg': acc_rel_avg_test, 'acc': acc_test, 'precision': pre_test, 'recall': rec_test, 'f1': f1_test,
            'acc of class': acc_rel_test, 'loss': loss_test.mean().item()}


def entry_batch_test(model_paths, device):
    each_class_acc_dicts = []
    result_dicts = []
    for path in model_paths:
        _cfg = get_cfg_defaults()
        _cfg.merge_from_file(os.path.join(path, 'config.yaml'))
        _cfg.merge_from_list(cmd_args.opts)
        _cfg.freeze()

        utils.set_seed(_cfg.EXP.SEED)

        model_path = os.path.join(path, 'model_best.pth')
        test_results = entry_test(_cfg, model_path, device)
        each_class_acc_dicts.append(test_results.pop('acc of class'))
        result_dicts.append(test_results)

    each_class_accuracies = merge_dicts(each_class_acc_dicts)
    final_result = merge_dicts(result_dicts)

    print()
    print("\U0001F680\U0001F680\U0001F680 Average Results over all models: \U0001F680\U0001F680\U0001F680")
    for predicate in each_class_accuracies:
        print('{} : {:.4f}\pm{:.4f}'.format(predicate, np.mean(each_class_accuracies[predicate]).item(), np.std(each_class_accuracies[predicate]).item()))

    print("\U0001F973 Accuracy of each predicate (in Latex format):")
    predicate_names = list(each_class_accuracies.keys())
    predicate_names = sorted(predicate_names)
    predicate_names = [predicate_names[i:i+5] for i in range(0, len(predicate_names), 5)]
    for name_list in predicate_names:
        print(" & ".join(name_list))
        for name in name_list:
            print("${:.2f}\pm{:.2f}$ & ".format(100 * np.mean(each_class_accuracies[name]).item(),
                                                100 * np.std(each_class_accuracies[name]).item()), end='')
        print()

    print("\U0001F60E Overall Results (in Latex format):")
    metrics = ['acc avg', 'acc', 'precision', 'recall', 'f1', 'loss']
    for m in metrics:
        print('${:.4f}\pm{:.4f}$ & '.format(np.mean(final_result[m]).item(), np.std(final_result[m]).item()), end=' ')
    print()


if __name__ == '__main__':
    DEVICE = torch.device('cuda')

    parser = argparse.ArgumentParser()
    parser.set_defaults(entry=lambda cmd_args: parser.print_help())
    parser.add_argument('--entry', type=str, default="train")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument('--exp-config', type=str, default="")
    parser.add_argument('--model-path', type=str, nargs='+', default="")
    parser.add_argument('--record-file', type=str, default="")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    cmd_args = parser.parse_args()

    if cmd_args.entry == "train":
        assert not cmd_args.exp_config == ""

        _cfg = get_cfg_defaults()
        _cfg.merge_from_file(cmd_args.exp_config)
        _cfg.merge_from_list(cmd_args.opts)
        if _cfg.EXP.EXP_ID == "":
            _cfg.EXP.EXP_ID = str(datetime.now())[:-7].replace(' ', '-')
        _cfg.freeze()

        utils.init_distributed_mode(cmd_args.local_rank)
        utils.set_seed(_cfg.EXP.SEED)
        DEVICE = torch.device("cuda", utils.get_rank())

        print(_cfg)

        entry_train(_cfg, DEVICE, cmd_args.record_file)

    elif cmd_args.entry == "test":
        assert not cmd_args.exp_config == ""
        assert len(cmd_args.model_path) == 1, "Only one model path is allowed for test flag"

        _cfg = get_cfg_defaults()
        _cfg.merge_from_file(cmd_args.exp_config)
        _cfg.merge_from_list(cmd_args.opts)
        _cfg.freeze()
        print(_cfg)

        utils.set_seed(_cfg.EXP.SEED)
        entry_test(_cfg, cmd_args.model_path, DEVICE)

    elif cmd_args.entry == "batch-test":
        assert len(cmd_args.model_path) > 1, "At least one model path is required for batch-test flag"
        entry_batch_test(cmd_args.model_path, DEVICE)

    else:
        raise ValueError(f"Invalid entry: {cmd_args.entry}")