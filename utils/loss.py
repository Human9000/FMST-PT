import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        smooth = 1
        dice = 0.
        # dice系数的定义
        for i in range(pred.size(1)):
            # 2 * p * t/ ( p*p + t*t + 1 )
            dice += 2 * (pred[:, i] * target[:, i]).sum(dim=1).sum(dim=1).sum(dim=1) / (pred[:, i].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) +
                                                                                        target[:, i].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) + smooth)

        # 返回的是dice距离
        dice = dice / pred.size(1)
        return torch.clamp((1 - dice).mean(), 0, 1)


class JaccardLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):

        smooth = 1

        # jaccard系数的定义
        jaccard = 0.

        for i in range(pred.size(1)):
            jaccard += (pred[:, i] * target[:, i]).sum(dim=1).sum(dim=1).sum(dim=1) / (pred[:, i].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) +
                                                                                       target[:, i].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) - (pred[:, i] * target[:, i]).sum(dim=1).sum(dim=1).sum(dim=1) + smooth)

        # 返回的是jaccard距离
        jaccard = jaccard / pred.size(1)
        return torch.clamp((1 - jaccard).mean(), 0, 1)


class TverskyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        smooth = 1
        dice = 0.

        def like(a, b):
            return (a * b).sum(dim=1).sum(dim=1).sum(dim=1)

        for i in range(pred.size(1)):
            # 内部准确率
            dice += like(pred[:, i], target[:, i]) / (like(pred[:, i], target[:, i]) +
                                                      0.3 * like(pred[:, i], (1 - target[:, i])) +
                                                      0.7 * like((1 - pred[:, i]), target[:, i]) + smooth)

        dice = dice / pred.size(1)

        return torch.clamp((1 - dice).mean(), 0, 2)


class AttnLoss(nn.Module):
    def __init__(self, abs=0.1):
        super().__init__()
        self.abs = abs

    def forward(self, pred, target):
        l1 = torch.abs(pred - target).mean()
        dice = 0.
        smooth = 1.
        for i in range(pred.size(1)):
            pi = pred[:, i]
            ti = target[:, i]
            inter = (pi * ti).sum(dim=[1, 2, 3])
            uninter1 = (pi + ti).sum(dim=[1, 2, 3])
            uninter2 = (pi**2 + ti**2).sum(dim=[1, 2, 3]) 
            dice += (smooth + inter)/(smooth + uninter1) \
                + (smooth + inter)/(smooth + uninter2)
        l2 = (1 - dice / pred.size(1)).mean()

        return self.abs*l1 * 10 + (1-self.abs)*l2 * 10
