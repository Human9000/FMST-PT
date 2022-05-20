import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import SimpleITK as sitk
from tqdm import tqdm


DEBUG = True


class CTDataset(Dataset):
    def __init__(self,
                 transforms=None,
                 interpolate=(128,128,128),
                 ATTN=False,
                 data_path='dataset/lits2017.json'
                 ):
        if os.path.isfile(data_path):  # 有没有数据配置结果
            print('===数据配置存在,加载中===')
            with open(data_path, 'rb') as fp:
                info = json.load(fp)
        else:  # 调用数据配置函数
            print('===缺少数据配置===')
            exit(0)
        self.info = info
        '''
{
    "name": "lits2017",
    "labels": {
        "0": "background",
        "1": "liver",
        "2": "tumer"
    },
    "aim_label": "1",
    "modality": {
        "0": "CT"
    },
    "numTest": 70,
    "numTraining": 120,
    "tensorImageSize": "3D",
    "test": [
        "/media/wgh/w2/data/LITS2017/Test_Data/test-volume-0.nii",
        "/media/wgh/w2/data/LITS2017/Test_Data/test-volume-69.nii"
    ],
    "train": [
        {
            "image": "/media/wgh/w1/data/LITS/TrainingData/volume-0.nii",
            "label": "/media/wgh/w1/data/LITS/TrainingData/segmentation-0.nii"
        },
        {
            "image": "/media/wgh/w1/data/LITS/TrainingData/volume-100.nii",
            "label": "/media/wgh/w1/data/LITS/TrainingData/segmentation-100.nii"
        }
    ],
    "valid": [
        {
            "image": "/media/wgh/w1/data/LITS/TrainingData/volume-101.nii",
            "label": "/media/wgh/w1/data/LITS/TrainingData/segmentation-101.nii"
        },
        {
            "image": "/media/wgh/w1/data/LITS/TrainingData/volume-130.nii",
            "label": "/media/wgh/w1/data/LITS/TrainingData/segmentation-130.nii"
        }
    ]
}'''
        self.transforms = transforms
        self.interpolate = interpolate
        self.ATTN = ATTN
        self.device = torch.device('cuda')
        self.dataset = self.__data_preload__()

    def __len__(self):
        return len(self.dataset)

    
    def readTrainFile(self, img_path, lab_path):  # 读数据文件
        img = sitk.ReadImage(img_path)
        lab = sitk.ReadImage(lab_path)

        x = sitk.GetArrayFromImage(img)
        y = sitk.GetArrayFromImage(lab) == int(self.info['aim_label'])
        index = np.where(y)
        foreground = x[y]
        conf = {
            'mu': np.mean(foreground),
            'theta': np.std(foreground),
            'window_l': np.min(index, axis=-1), 
            'window_r': np.max(index, axis=-1),
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
        attn = (attn + F.max_pool3d(attn, 3, 1, 1) )/ 2
        attn = F.avg_pool3d(attn, 7, 1, 3)
        attn =  F.interpolate(attn, size=y.shape[2:], mode='trilinear', align_corners=True)
        
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

    def __data_preload__(self):
        spectrum = []
        tr = []
        val = []
        te = []

        print('Data loading ...')
        for path in tqdm(self.info['train'], desc='train', total=len(self.info['train'])):
            item = self.readTrainFile(path['image'], path['label'])
            mu, theta = item['conf']['mu'], item['conf']['theta']
            spectrum.extend([mu + theta, mu - theta])
            tr.append(item)

        for path in tqdm(self.info['valid'], desc='valid', total=len(self.info['valid'])):
            item = self.readTrainFile(path['image'], path['label'])
            mu, theta = item['conf']['mu'], item['conf']['theta']
            spectrum.extend([mu + theta, mu - theta])
            val.append(item)

        dataset = {
            'train': tr,
            'valid': val,
            'test': te,
            'mean': np.mean(spectrum),
            'std': np.mean(spectrum),
        }
        return dataset

    def __getitem__(self, index):
        item = self.dataset[index]
        if self.transforms:
            x,y = self.transforms(x,y)
        return item


def main():
    ct = CTDataset(data_path='lits2017.json', interpolate=(128,128,128))


if __name__ == '__main__':
    main()
