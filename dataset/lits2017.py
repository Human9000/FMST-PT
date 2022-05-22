import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import SimpleITK as sitk
from tqdm import tqdm
from scipy.stats import norm
from transforms import Compose, RandomCrop,  RandomFlip, RandomRotate, RandomTranspose,Mapping


class DataAnalysis():
    def __init__(self,
                 interpolate=(128, 128, 128),
                 ATTN=True,
                 data_conf='dataset/lits2017.json') -> None:

        with open(data_conf, 'rb') as fp:
            info = json.load(fp)

        self.info = info
        self.interpolate = interpolate
        self.ATTN = ATTN
        self.device = torch.device('cuda')
        self.dataset = self.__data_preload__()

    def readTrainFile(self, img_path, lab_path):  # 读数据文件
        img = sitk.ReadImage(img_path)
        lab = sitk.ReadImage(lab_path)

        x = sitk.GetArrayFromImage(img)
        y = sitk.GetArrayFromImage(lab) == int(self.info['aim_label'])
        # index = np.where(y)
        foreground = x[y]
        conf = {
            'mu': np.mean(foreground),
            'theta': np.std(foreground),
            'Direction': img.GetDirection(),
            'Spacing': img.GetSpacing(),
            'Origin': img.GetOrigin(),
            'Shape': x.shape,
        }

        # GPU 操作，torch.Tensor.GPU
        x = torch.FloatTensor(x).to(self.device).unsqueeze(0).unsqueeze(0)
        y = torch.FloatTensor(y).to(self.device).unsqueeze(0).unsqueeze(0)

        x = F.interpolate(x, size=self.interpolate)
        y = F.interpolate(y, size=self.interpolate)

        attn = F.interpolate(1+y, size=(32, 32, 32)) / 2
        attn = (attn + F.max_pool3d(attn, 3, 1, 1)) / 2
        attn = F.avg_pool3d(attn, 7, 1, 3)
        attn = F.interpolate(
            attn, size=y.shape[2:], mode='trilinear', align_corners=True)

        # 返回 CPU，torch.Tensor.CPU
        x = x.cpu().squeeze(0).squeeze(0)
        y = y.cpu().squeeze(0).squeeze(0)
        attn = attn.cpu().squeeze(0).squeeze(0)
        torch.cuda.empty_cache()

        item = {
            'image': x,
            'label': y,
            'attn': attn,
            'conf': conf
        }

        return item

    def readTestFile(self, img_path):  # 读数据文件
        img = sitk.ReadImage(img_path)

        x = sitk.GetArrayFromImage(img)

        conf = {
            'mu': None,
            'theta': None,
            'Direction': img.GetDirection(),
            'Spacing': img.GetSpacing(),
            'Origin': img.GetOrigin(),
            'Shape': x.shape,
        }

        # GPU 操作，torch.Tensor.GPU
        x = torch.FloatTensor(x).to(self.device).unsqueeze(0).unsqueeze(0)
        x = F.interpolate(x, size=self.interpolate)
        # 返回 CPU，torch.Tensor.CPU
        x = x.cpu().squeeze(0).squeeze(0)
        torch.cuda.empty_cache()

        item = {
            'image': x,
            'label': None,
            'attn': None,
            'conf': conf
        }

        return item

    def __data_preload__(self):
        spectrum = []
        tr = []
        val = []
        te = []

        print('\033[34;0mData Analysis  ... \033[0m')
        for path in tqdm(self.info['train'], desc='train', total=len(self.info['train'])):
            item = self.readTrainFile(path['image'], path['label'])
            mu, theta = item['conf']['mu'], item['conf']['theta']
            spectrum.extend([mu + theta, mu - theta])
            tr.append(item)
            break

        for path in tqdm(self.info['valid'], desc='valid', total=len(self.info['valid'])):
            item = self.readTrainFile(path['image'], path['label'])
            mu, theta = item['conf']['mu'], item['conf']['theta']
            spectrum.extend([mu + theta, mu - theta])
            val.append(item)
            break

        for path in tqdm(self.info['test'], desc='test', total=len(self.info['test'])):
            item = self.readTestFile(path)
            te.append(item)
            break

        mu = np.mean(spectrum)
        theta = np.std(spectrum)

        # [1%, 99%] 置信度
        l = norm.ppf(0.01) * theta + mu
        r = norm.isf(0.01) * theta + mu
        print(l, r)
        for i in tr:
            i['image'] = (torch.clamp(i['image'], l, r) - mu) / theta
        for i in val:
            i['image'] = (torch.clamp(i['image'], l, r) - mu) / theta
        for i in te:
            i['image'] = (torch.clamp(i['image'], l, r) - mu) / theta

        print(tr[0]['image'])
        
        dataset = {
            'train': tr,
            'valid': val,
            'test': te,
        }
        return dataset


class CTDataset(Dataset):
    def __init__(self,
                 transforms=None,
                 interpolate=(128, 128, 128),
                 ATTN=True,
                 data_conf='dataset/lits2017.json',
                 data_type='train',  # 'train', 'valid', 'test'
                 ):

        self.data_type = data_type
        self.transforms = transforms
        self.dataset = DataAnalysis(interpolate, ATTN, data_conf).dataset[data_type]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]

        if self.transforms:  # 数据增强
            item = self.transforms(item)

        return item


def main():
    ct = CTDataset(data_type='train',
                   data_conf='lits2017.json',
                   transforms=Compose(
                        RandomCrop(slices=96),  # 随机裁剪96个切片
                        RandomFlip(dims=[0, 1, 2]),  # 维度内依概率翻转默认50%概率
                        RandomRotate(max_angle=180),  # 最后两个维度随机旋转[0-180°]
                        RandomTranspose(dims=[0, 1, 2]),  # 依概率随机交换指定维度默认概率50%
                        Mapping(0,1), # 数据最小值和最大值映射为[0-1]
                   ))
    item = ct.__getitem__(0)
    print(item['image'])


if __name__ == '__main__':
    main()
