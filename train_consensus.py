import os
from dataset.lits2017 import DataAnalysis, Dataset
from dataset import transforms as T
from model.fmst import SNet
from utils import weights_init, metrics, loss, logger

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import OrderedDict


def tr_epoch(model, device, loader, optimizer, loss_func, metricsLoss, metricsDice, metricsDiceRes,metricsAttnDice):
    model.train()

    metricsLoss.reset()
    metricsDice.reset()
    metricsDiceRes.reset()
    metricsAttnDice.reset()

    for item in tqdm(loader, total=len(loader)):
        x = item['image'].to(device)
        y = item['label'].to(device)

        optimizer.zero_grad()  # 重置梯度
        output = model(x)  # 模型求解

        # 残差反向传播计算
        loss = loss_func(output[0], y)
        for i in range(len(output)-1, 0, -1):
            rrSidualC = 1.5 - 1/(1 + 2.7**(-i))  # 权重系数
            loss += rrSidualC * loss_func(output[i], y)

        loss.backward()  # 反向传播
        optimizer.step()  # 优化器更新

        metricsLoss.update(loss.item(), y.size(0))
        metricsDice.update(output[0], y)
        metricsAttnDice.update(output[0], y)
        metricsDiceRes.update(torch.mean(
            torch.stack(output, dim=-1), dim=-1), y)

    return OrderedDict({'loss': metricsLoss.avg,
                        'dice0': metricsDice.avg[0],
                        'dice1': metricsDice.avg[1],
                        'diceRes0': metricsDiceRes.avg[0],
                        'diceRes1': metricsDiceRes.avg[1],
                        'diceAttn0': metricsAttnDice.avg[0],
                        'diceAttn1': metricsAttnDice.avg[1],
                        })


def val_epoch(model, device, loader, loss_func, metricsLoss, metricsDice, metricsDiceRes,metricsAttnDice):
    model.eval()

    metricsLoss.reset()
    metricsDice.reset()
    metricsDiceRes.reset()
    metricsAttnDice.reset()

    for item in tqdm(loader, total=len(loader)):
        x = item['image'].to(device)
        y = item['label'].to(device)
        output = model(x)  # 模型求解
        save(torch.cat((x,x,x),dim=1),f"val/consensus/{item['conf']['name'][0]}_x.png")
        save(torch.cat((output[0][:,1:], output[0]),dim=1),f"val/consensus/{item['conf']['name'][0]}_pre.png")
        save(torch.cat((y[:,1:],y[:,1:],y[:,1:]),dim=1),f"val/consensus/{item['conf']['name'][0]}_y.png")
        loss = loss_func(output[0], y)
        for i in range(len(output)-1, 0, -1):
            rrSidualC = 1.5 - 1/(1 + 2.7**(-i))  # 权重系数
            loss += rrSidualC * loss_func(output[i], y)

        metricsLoss.update(loss.item(), y.size(0))
        metricsDice.update(output[0], y)
        metricsAttnDice.update(output[0], y)
        metricsDiceRes.update(torch.mean(
            torch.stack(output, dim=-1), dim=-1), y)

    return OrderedDict({'loss': metricsLoss.avg,
                        'dice0': metricsDice.avg[0],
                        'dice1': metricsDice.avg[1],
                        'diceRes0': metricsDiceRes.avg[0],
                        'diceRes1': metricsDiceRes.avg[1],
                        'diceAttn0': metricsAttnDice.avg[0],
                        'diceAttn1': metricsAttnDice.avg[1],
                        })

def save(img3d,# (1, 3, 128, 128, 128)
         file_path):
    import cv2 as cv
    from torch.nn import functional as F    
    from einops import rearrange
    img3d = F.interpolate(img3d, size=(64,64,64)) # (1,3,64,64,64)
    # 1,3,8*64,8*64
    img = rearrange(img3d, 'b c (d0 d1) w h -> b (d0 w) (d1 h) c', d0=8) # (1, 8*64,8*64, 3)
    img = img[0].cpu()
    img = (img - img.min())/(img.max()-img.min()) * 255
    img = img.detach().numpy()
     
    img = rearrange(img, '(d0 d1 w) (d2 d3 h) c ->(d0 w) (d2 h) c (d1 d3)', d0=4, d1=2, d2=4, d3=2)[...,0] # (1, 8*64,8*64, 3)
    cv.imwrite(file_path, img)
    

def main():
    device = torch.device('cuda')

    # 数据集配置
    analysis = DataAnalysis(interpolate=(128, 128, 128),
                            ATTN=True,
                            data_conf='dataset/lits2017.json',
                            data_pkl='/media/wgh/u2/lits2017.pkl',
                            NEWPKL=False, # False:如果有pkl数据直接读取,True：生成新的pkl文件
                            )
    
    
    tr_dataset = Dataset(
        analysis,
        data_type='train',
        transforms=T.Compose(
            T.CopyTo(device), # 复制到显卡上
            T.NormalDirection(),
            # T.RandomCrop(slices=96, dims=[0]),  # 随机裁剪96个切片
            # T.RandomFlip(dims=[1, 2]),  # 维度内依概率翻转默认50%概率
            # T.RandomRotate(max_angle=10),  # 最后两个维度随机旋转[0-30°]
            # T.RandomTranspose(dims=[0, 1, 2]),  # 依概率随机交换指定维度默认概率50%
            T.Mapping(0, 1),  # 数据最小值和最大值映射为[0-1]
            T.AddDim(0),  # 添加通道维度
            T.OneHot2classs(),  # 通道维度做成onehot编码
        ))
    val_dataset = Dataset(analysis,
                          data_type='valid',
                          transforms=T.Compose(
                              T.CopyTo(device), # 复制到显卡上
                              T.NormalDirection(), # 统一方向
                              T.Mapping(0, 1),  # 数据最小值和最大值映射为[0-1]
                              T.AddDim(0),  # 添加通道维度
                              T.OneHot2classs())  # 通道维度做成onehot编码
                          )
    # 模型配置
    save_path = 'save/consensus/'

    if not os.path.isdir(save_path):
        os.makedirs(save_path)
        
    model = SNet().to(device)
    # torch.nn.Module.load_state_dict()
    model.apply(weights_init.init_model)
    ckpt = torch.load(os.path.join(save_path, 'init_sunet.pth'))
    model.load_state_dict(ckpt['net'],strict=False )
    
    # 超参数配置
    loss_func = loss.AttnLoss(0.1)  # 损失函数
    # loss_func = loss.TverskyLoss()  # 损失函数
    metricsLoss = metrics.LossAverage()  # 损失函数评价
    metricsDice = metrics.DiceAverage(2)  # 2分类dice系数评价
    metricsDiceRes = metrics.DiceAverage(2)  # 2分类dice系数评价
    metricsAttnDice = metrics.AttnDiceAverage(2)  # 2分类dice系数评价
    tr_loader = DataLoader(dataset=tr_dataset, shuffle=True)   # 训练数据集读取器
    val_loader = DataLoader(dataset=val_dataset, shuffle=False)   # 检验数据集读取器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 优化器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.2, patience=4, verbose=True)  # 学习率调整器


    log = logger.Train_Logger(save_path, "train_log")  # 日志
    best = {'epoch': -1, 'log': None}

    for epoch in range(2000):

        print("=======Epoch:{}=======lr:{}".format(
            epoch, optimizer.state_dict()['param_groups'][0]['lr']))
        print('Best performance at Epoch: {} | {}'.format(
            best['epoch'], best['log']))
        print("==="*8)
        torch.cuda.empty_cache()

        tr_log = tr_epoch(model, device, tr_loader, optimizer,
                          loss_func, metricsLoss, metricsDice, metricsDiceRes, metricsAttnDice)
        val_log = val_epoch(model, device, val_loader,
                            loss_func, metricsLoss, metricsDice, metricsDiceRes, metricsAttnDice)

        # 日志
        log.update(epoch, tr_log, val_log)

        scheduler.step(val_log['loss'])

        state = {'net': model.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'epoch': epoch}

        # 保存当前权重
        torch.save(state, os.path.join(save_path, 'latest_model.pth'))  # 保存权重

        if (best['log'] is None) or \
                (val_log['loss'] < best['log']['loss']):  # 保存最优权重
            print('Saving best model')
            torch.save(state, os.path.join(save_path, 'best_model.pth'))
            best['epoch'] = epoch
            best['log'] = val_log

        # 提前结束
        if epoch - best['epoch'] >= 20:
            break


if __name__ == '__main__':
    main()

