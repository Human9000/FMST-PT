import random
from torchvision.transforms.functional import rotate,InterpolationMode


class RandomCrop:
    def __init__(self, slices):
        self.slices = slices

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
            l, r = self._get_range(item["image"].size(dim), self.slices)
            arg.extend([l,r])
        
        item['image'] = self.__do__(item['image'], *arg)
        item['label'] = self.__do__(item['label'], *arg)
        item['attn'] = self.__do__(item['attn'], *arg)
        
        return item


class RandomFlip:
    def __init__(self, prob=0.5, dims=[0,1,2]):
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
        img = rotate(img - full_color, angle, InterpolationMode.BILINEAR) + full_color
        return img

    def __call__(self, item):
        angle = random.randint(0, self.max_angle)
        item['image'] = self._rotate(item['image'], angle)
        item['label'] = self._rotate(item['label'], angle)
        item['attn'] = self._rotate(item['attn'], angle)
        return item


class Mapping:
    def __init__(self, left=0, right = 1):
        self.l = left
        self.r = right

    def _do(self, img):
        m_in =  img.min()
        d = img.max() - m_in
        img = (img - m_in) / d * (self.r - self.l) + self.l 
        return img

    def __call__(self, item):
        item['image'] = self._do(item['image'])
        return item


class Compose:
    def __init__(self, *transforms):
        self.transforms = transforms

    def __call__(self, item):
        for t in self.transforms:
            item = t(item)
        return item
