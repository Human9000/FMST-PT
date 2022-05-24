import random
from torchvision.transforms.functional import rotate, InterpolationMode
import torch
from torch.nn import functional as F
from inference_consensus import inference_attn_window
from inference_attention import inference_seg


class RandomCrop:
    def __init__(self, slices, dims=[0,1,2]):
        self.slices = slices
        self.dims = dims

    def _get_range(self, slices, crop_slices):
        if slices < crop_slices:
            start = 0
        else:
            start = random.randint(0, slices - crop_slices)
        end = start + crop_slices
        if end > slices:
            end = slices
        return start, end

    def __do__(self, x, l0, r0, l1, r1, l2, r2):
        if x is None:
            return None

        return x[..., l0:r0, l1:r1, l2:r2]

    def __call__(self, item):
        arg = []
        for dim in range(len(item["image"].shape)):
            if dim in self.dims:
                l, r = self._get_range(item["image"].size(dim), self.slices)
            else:
                l,r = 0, item["image"].size(dim)
            arg.extend([l, r])
            
        item['image'] = self.__do__(item['image'], *arg)
        item['label'] = self.__do__(item['label'], *arg)
        item['attn'] = self.__do__(item['attn'], *arg)

        return item


class RandomFlip:
    def __init__(self, prob=0.5, dims=[0, 1, 2]):
        self.prob = prob
        self.dims = dims

    def _flip(self, img, prob):
        if img is None:
            return None

        if prob <= self.prob:
            for dim in self.dims:
                img = img.flip(dim)
        return img

    def __call__(self, item):
        prob = random.uniform(0, 1)
        item['image'] = self._flip(item['image'], prob)
        item['label'] = self._flip(item['label'], prob)
        item['attn'] = self._flip(item['attn'], prob)
        return item


class RandomTranspose:
    def __init__(self, prob=0.5,
                 dims=[0, 1, 2],
                 ):
        self.prob = prob
        self.dims = dims

    def _do(self, img, prob):
        if img is None:
            return None

        if prob <= self.prob:
            dims = self.dims
            img = img.permute(*dims)
        return img

    def __call__(self, item):
        prob = random.uniform(0, 1)
        random.shuffle(self.dims)

        item['image'] = self._do(item['image'], prob)
        item['label'] = self._do(item['label'], prob)
        item['attn'] = self._do(item['attn'], prob)

        return item


class RandomRotate:
    def __init__(self, max_angle=180):
        self.max_angle = max_angle

    def _rotate(self, img, angle):
        if img is None:
            return None

        full_color = img.min()
        img = rotate(img - full_color, angle,
                     InterpolationMode.BILINEAR) + full_color
        return img

    def __call__(self, item):
        angle = random.randint(0, self.max_angle)
        item['image'] = self._rotate(item['image'], angle)
        item['label'] = self._rotate(item['label'], angle)
        item['attn'] = self._rotate(item['attn'], angle)
        return item


class Mapping:
    def __init__(self, left=0, right=1):
        self.l = left
        self.r = right

    def _do(self, img):
        m_in = img.min()
        d = img.max() - m_in
        img = (img - m_in) / d * (self.r - self.l) + self.l
        return img

    def __call__(self, item):
        item['image'] = self._do(item['image'])
        return item


class CopyTo:
    def __init__(self, device):
        self.device = device

    def __call__(self, item:dict):
        new_items = {}
        for key, value in item.items():
            new_items[key] = value
        new_items['image'] = item['image'].to(self.device)
        new_items['label'] = item['label'].to(self.device)
        new_items['attn'] = item['attn'].to(self.device)
        return new_items


class Interpolate:
    def __init__(self, size=(128,128,128)):
        self.size = size
        
    def _do(self, x, mode='trilinear'):
        xin = x.unsequenze(0).unsequenze(0)
        y = F.interpolate(xin, self.size, mode=mode, align_corners=True)
        return y.sequenze(0).sequenze(0)
    
    def __call__(self, item):
        item['image']= self._do(item['image'])
        # item['label']= self._do(item['label'], 'nearest')
        # item['attn']= self._do(item['attn'], 'nearest')
        item['label']= self._do(item['label'])
        item['attn']= self._do(item['attn'])
        return item

class AddDim:
    def __init__(self, newdim=0) -> None:
        self.newdim = newdim

    def __call__(self, item):
        item['image'] = item['image'].unsqueeze(self.newdim)
        item['label'] = item['label'].unsqueeze(self.newdim)
        item['attn'] = item['attn'].unsqueeze(self.newdim)

        return item


class NormalDirection:
    def __init__(self, newdim=0) -> None:
        self.newdim = newdim
    
    def _do(self, x, direction):
        if direction[0] < 0:
            x = x.flip(0)
        if direction[4] < 0:
            x = x.flip(1)
        if direction[8] < 0:
            x = x.flip(2)
        return x

    def __call__(self, item):
        item['image'] = self._do(item['image'], item['conf']['Direction'])
        item['label'] = self._do(item['label'], item['conf']['Direction'])
        item['attn'] = self._do(item['attn'], item['conf']['Direction'])

        return item


class Consensus:
    def __init__(self, newdim=0) -> None:
        self.newdim = newdim
    
    def _do(self, x, window):
        [x0,x1,y0,y1,z0,z1] = window
        # print(window, x.shape)
        x = x[x0:x1, y0:y1, z0:z1]
        
        x = x.unsqueeze(0).unsqueeze(0)
        
        y = F.interpolate(x, size=(128,128,128), mode='trilinear', align_corners=True)    
        y = y.squeeze(0).squeeze(0)# 返回 CPU，torch.Tensor.CPU
        
        return y
 
    def __call__(self, item):
        window, attn = inference_attn_window(item)
        item['image'] = self._do(item['image'], window)
        item['label'] = self._do(item['label'], window)
        item['attn'] = self._do(attn, window)
        
        item['window'] = window        
        
        return item
    

class Attention:
    def __init__(self, newdim=0) -> None:
        self.newdim = newdim

    def __call__(self, item):
        item['seg'] = inference_seg(item) 
        return item
    

class OneHot2classs:
    def _do(self, y):
        return torch.cat((1-y, y), dim=0)

    def __call__(self, item):
        item['label'] = self._do(item['label'])
        item['attn'] = self._do(item['attn'])
        return item


class Compose:
    def __init__(self, *transforms):
        self.transforms = transforms

    def __call__(self, item):
        for t in self.transforms:
            item = t(item)
        return item
