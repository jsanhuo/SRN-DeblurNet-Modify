import torch
import torch.nn as nn
import torchvision
from network import SRNDeblurNet
import numpy as np
from log import TensorBoardX
from utils import *
import train_config as config
from data import Dataset
from functools import partial
from tqdm import tqdm
from time import time
import copy
import sys
from metric import PSNR,SSIM
log10 = np.log(10)
MAX_DIFF = 2

p_s = PSNR()
s_s = SSIM()

# 计算损失
def compute_loss(db256, db128, db64, batch):
    assert db256.shape[0] == batch['label256'].shape[0]

    loss = 0
    loss += mse(db256, batch['label256'])
    # psnr = 10 * torch.log(MAX_DIFF ** 2 / loss) / log10
    loss += mse(db128, batch['label128'])
    loss += mse(db64, batch['label64'])
    # 计算psnr
    psnr = p_s(batch['label256'],db256)
    # 计算ssim
    ssim = s_s(batch['label256'],db256)
    # 返回mse，psnr，ssim
    return {'mse': loss, 'psnr': psnr,'ssim':ssim}


# 反向传播
def backward(loss, optimizer):
    optimizer.zero_grad()
    loss['mse'].backward()
    # 梯度裁剪 对convlstm进行裁剪
    torch.nn.utils.clip_grad_norm_(net.module.convlstm.parameters(), 3)
    optimizer.step()
    return

# 设置学习率
def set_learning_rate(optimizer, epoch):
    optimizer.param_groups[0]['lr'] = config.train['learning_rate'] * 0.3 ** (epoch // 500)


if __name__ == "__main__":
    # TensorBoardX 路径
    tb = TensorBoardX(config_filename='train_config.py', sub_dir=config.train['sub_dir'])
    log_file = open('{}/{}'.format(tb.path, 'train.log'), 'w')

    train_img_list = open(config.train['train_img_list'], 'r').read().strip().split('\n')
    val_img_list = open(config.train['val_img_list'], 'r').read().strip().split('\n')
    # 数据加载
    train_dataset = Dataset(train_img_list)
    val_dataset = Dataset(val_img_list)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train['batch_size'], shuffle=True,
                                                   drop_last=True, num_workers=8, pin_memory=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=config.train['val_batch_size'], shuffle=True,
                                                 drop_last=True, num_workers=2, pin_memory=True)
    # 损失函数
    mse = torch.nn.MSELoss().cuda()
    # 定义网络
    net = torch.nn.DataParallel(SRNDeblurNet(xavier_init_all=config.net['xavier_init_all'])).cuda()
    # net = SRNDeblurNet(xavier_init_all = config.net['xavier_init_all']).cuda()

    # Adam和SGD两种优化方式
    assert config.train['optimizer'] in ['Adam', 'SGD']
    if config.train['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=config.train['learning_rate'],
                                     weight_decay=config.loss['weight_l2_reg'])
    if config.train['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=config.train['learning_rate'],
                                    weight_decay=config.loss['weight_l2_reg'], momentum=config.train['momentum'],
                                    nesterov=config.train['nesterov'])
    # 断点续训练
    last_epoch = -1

    if config.train['resume'] is not None:
        last_epoch = load_model(net, config.train['resume'], epoch=config.train['resume_epoch'])

    if config.train['resume_optimizer'] is not None:
        _ = load_optimizer(optimizer, net, config.train['resume_optimizer'], epoch=config.train['resume_epoch'])
        assert last_epoch == _

    # train_loss_epoch_list = []

    train_loss_log_list = []
    val_loss_log_list = []
    first_val = True

    t = time()
    # convlstm_params = net.module.convlstm.parameters()
    # net_params = net.module.parameters()
    best_val_psnr = 0
    best_net = None
    best_optimizer = None
    for epoch in tqdm(range(last_epoch + 1, config.train['num_epochs']), file=sys.stdout):
        # 设置学习率
        set_learning_rate(optimizer, epoch)
        # 在tensorboard里面添加lr
        tb.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch, 'train')
        # tloss = 0
        # tpsnr = 0
        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), file=sys.stdout,
                                desc='training'):
            t_list = []
            for k in batch:
                batch[k] = batch[k].cuda()
                batch[k].requires_grad = False
            # 正向传播
            db256, db128, db64 = net(batch['img256'], batch['img128'], batch['img64'])
            # 计算损失
            loss = compute_loss(db256, db128, db64, batch)
            # 反向传播
            backward(loss, optimizer)

            for k in loss:
                loss[k] = float(loss[k].cpu().detach().numpy())
            train_loss_log_list.append({k: loss[k] for k in loss})
        # 计算训练集的mse和psnr均值
        train_loss_log_dict = {k: float(np.mean([dic[k] for dic in train_loss_log_list])) for k in
                               train_loss_log_list[0]}
        # 写入tensorboard
        for k, v in train_loss_log_dict.items():
            tb.add_scalar(k, v, epoch, 'train')
        # vloss = 0
        # vpsnr = 0
        # validate and log 验证
        if first_val or epoch % config.train['log_epoch'] == config.train['log_epoch'] - 1:
            with torch.no_grad():
                first_val = False
                for step, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader), file=sys.stdout,
                                        desc='validating'):
                    for k in batch:
                        batch[k] = batch[k].cuda()
                        batch[k].requires_grad = False
                    # 正向传播
                    db256, db128, db64 = net(batch['img256'], batch['img128'], batch['img64'])
                    # 计算损失
                    loss = compute_loss(db256, db128, db64, batch)
                    # 将值变为float 便于计算
                    for k in loss:
                        loss[k] = float(loss[k].cpu().detach().numpy())
                    # 添加进val_loss_log_list内记录
                    val_loss_log_list.append({k: loss[k] for k in loss})
                    # vloss += loss['mse']
                    # vpsnr += loss['psnr']
                # train_loss_log_dict = {k: float(np.mean([dic[k] for dic in train_loss_log_list])) for k in
                #                        train_loss_log_list[0]}
                # 求平均val_loss_log_dict 为 mse和psnr的均值
                val_loss_log_dict = {k: float(np.mean([dic[k] for dic in val_loss_log_list])) for k in
                                     val_loss_log_list[0]}

                # 向tensorboard里面写入
                for k, v in val_loss_log_dict.items():
                    tb.add_scalar(k, v, epoch, 'val')
                # tb.add_scalar('mse', vloss / len(val_dataloader), epoch, 'val')
                # tb.add_scalar('psnr', vpsnr / len(val_dataloader), epoch, 'val')
                if best_val_psnr < val_loss_log_dict['psnr']:
                    best_val_psnr = val_loss_log_dict['psnr']
                    save_model(net, tb.path, epoch)
                    save_optimizer(optimizer, net, tb.path, epoch)
                # 清空集合
                train_loss_log_list.clear()
                val_loss_log_list.clear()
                # 获得时间
                tt = time()
                # 拼接日志消息
                log_msg = ""
                log_msg += "epoch {} , {:.2f} imgs/s".format(epoch, (
                            config.train['log_epoch'] * len(train_dataloader) * config.train['batch_size'] + len(
                        val_dataloader) * config.train['val_batch_size']) / (tt - t))

                log_msg += " | train : "
                # 将训练集信息写入
                for idx, k_v in enumerate(train_loss_log_dict.items()):
                    k, v = k_v
                    log_msg += "{} {:.5f} {}".format(k, v, ',')
                log_msg += "  | val : "
                # 将验证机消息写入
                for idx, k_v in enumerate(val_loss_log_dict.items()):
                    k, v = k_v
                    log_msg += "{} {:.5f} {}".format(k, v, ',')
                # 将消息输出到控制台
                tqdm.write(log_msg, file=sys.stdout)
                sys.stdout.flush()
                # 写入日志文件
                log_file.write(log_msg + '\n')
                log_file.flush()
                t = time()
                # print( torch.max( predicts , 1  )[1][:5] )

            # train_loss_epoch_list = []
