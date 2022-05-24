from model.fmst import AttnNet

import torch

device = torch.device('cuda')

def attention_conf():
    model = AttnNet().to(device)
    ckpt = torch.load('save/attention/best_attnnet.pth')
    model.load_state_dict(ckpt['net'], strict=False)
    return model

attention_model = attention_conf()

def inference_seg(item):
    attention_model.eval()
    x = item['image']
    attn = item['attn']
    xin = x.unsqueeze(0).unsqueeze(0).to(device)
    attn = attn.unsqueeze(0).unsqueeze(0).to(device)
    pre_y = attention_model(xin, attn)[0]  # 模型求解
    return pre_y.squeeze(0)
