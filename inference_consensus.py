import os

from model.fmst import SNet

import torch
from torch.nn import functional as F
import numpy as np

device = torch.device('cuda')

def consensus_conf():
    model = SNet().to(device)
    save_path = 'save/consensus/'
    ckpt = torch.load(os.path.join(save_path, 'best_sunet.pth'))
    model.load_state_dict(ckpt['net'], strict=False)
    return model

consensus_model = consensus_conf()

def inference_attn_window(item):
    consensus_model.eval()    
    # print(xin.shape)
    x = item['image']
    xin = x.unsqueeze(0).unsqueeze(0).to(device)
    pre_y = consensus_model(xin)[0]  # 模型求解
    
    attn = F.interpolate(pre_y[:, 1:], size=(32, 32, 32))
    attn = (attn + F.max_pool3d(attn, 3, 1, 1)) / 2
    attn = F.avg_pool3d(attn, 7, 1, 3)
    attn = F.interpolate(attn, size=xin.shape[2:], mode='trilinear', align_corners=True)
    attn = (0.1 + attn) / 1.1
    attn = attn.cpu().squeeze(0).squeeze(0)
    
    
    y = F.interpolate(pre_y[:, :1], size=(32, 32, 32)) # 背景重采样
    y = F.max_pool3d(y, 3, 1, 1)  # 背景最大池化
    y = F.max_pool3d(1-y, 5, 1, 2) # 前景最大池化
    y = F.interpolate(y, size=xin.shape[2:], mode='trilinear', align_corners=True)    
    y = y.cpu().squeeze(0).squeeze(0)# 返回 CPU，torch.Tensor.CPU
    
    try:
        index = np.where(y>0)
        x0,y0,z0 = np.min(index,axis=-1) 
        x1,y1,z1 = np.max(index,axis=-1) + 1
    except:
        save(torch.cat((xin, xin, xin), dim=1), f"{item['conf']['name']}x.png")
        save(torch.cat((pre_y[:,1:], pre_y[:,1:], pre_y[:,1:]), dim=1), f"{item['conf']['name']}pre.png")
        exit()
    
    return [x0,x1,y0,y1,z0,z1], attn 


def save(img3d,  # (1, 3, 128, 128, 128)
         file_path):
    import cv2 as cv
    from torch.nn import functional as F
    from einops import rearrange
    img3d = F.interpolate(img3d, size=(64, 64, 64))  # (1,3,64,64,64)
    # 1,3,8*64,8*64
    # (1, 8*64,8*64, 3)
    img = rearrange(img3d, 'b c (d0 d1) w h -> b (d0 w) (d1 h) c', d0=8)
    img = img[0].cpu()
    img = (img - img.min())/(img.max()-img.min()) * 255
    img = img.detach().numpy()

    img = rearrange(img, '(d0 d1 w) (d2 d3 h) c ->(d0 w) (d2 h) c (d1 d3)',
                    d0=4, d1=2, d2=4, d3=2)[..., 0]  # (1, 8*64,8*64, 3)
    cv.imwrite(file_path, img)
