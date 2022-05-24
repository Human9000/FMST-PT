import torch
import numpy as np


class LossAverage(object):
    """Computes and stores the average and current value for calculate average loss"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = round(self.sum / self.count, 4)


class DiceAverage(object):
    """Computes and stores the average and current value for calculate average loss"""

    def __init__(self, class_num=2):
        self.class_num = class_num
        self.reset()

    def reset(self):
        self.value = np.asarray([0]*self.class_num, dtype='float64')
        self.avg = np.asarray([0]*self.class_num, dtype='float64')
        self.sum = np.asarray([0]*self.class_num, dtype='float64')
        self.count = 0

    def update(self, logits, targets):
        self.value = DiceAverage.get_dices(logits, targets)
        self.sum += self.value
        self.count += 1
        self.avg = np.around(self.sum / self.count, 4)

    @staticmethod
    def get_dices(logits, targets):
        dices = []
        for class_index in range(targets.size()[1]):
            pi = logits[:, class_index]
            ti = targets[:, class_index]

            inter = torch.sum(pi * ti)
            uninter = torch.sum(pi) + torch.sum(ti)

            dice = (1 + 2 * inter) / (1 + uninter)

            dices.append(dice.item())
        return np.asarray(dices)


class AttnDiceAverage(object):
    """Computes and stores the average and current value for calculate average loss"""

    def __init__(self, class_num=2):
        self.class_num = class_num
        self.reset()

    def reset(self):
        self.value = np.asarray([0]*self.class_num, dtype='float64')
        self.avg = np.asarray([0]*self.class_num, dtype='float64')
        self.sum = np.asarray([0]*self.class_num, dtype='float64')
        self.count = 0

    def update(self, logits, targets):
        self.value = AttnDiceAverage.get_dices(logits, targets)
        self.sum += self.value
        self.count += 1
        self.avg = np.around(self.sum / self.count, 4)

    @staticmethod
    def get_dices(logits, targets):
        dices = []
        for class_index in range(targets.size()[1]):
            pi = logits[:, class_index]
            ti = targets[:, class_index]
            inter = torch.sum(pi * ti)
            uninter = torch.sum(pi**2) + torch.sum(ti**2)
            dice = (1 + 2 * inter) / (1 + uninter)

            dices.append(dice.item())
        return np.asarray(dices)
