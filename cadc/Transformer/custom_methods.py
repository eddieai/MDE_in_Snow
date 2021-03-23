import random
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import torch.nn as nn
from PIL import Image
from .base_methods import BaseMethod

"""
This file defines some transform examples.
Each transform method is defined by using BaseMethod class
"""


class ToPIL(BaseMethod):
    """
    Transform method to convert images as PIL Image.
    """

    def __init__(self):
        BaseMethod.__init__(self)
        self.to_pil = transforms.ToPILImage()

    def __call__(self, data_item):
        self.set_data(data_item)

        if not self._is_pil_image(self.img):
            data_item['img'] = self.to_pil(self.img)
        if not self._is_pil_image(self.depth):
            data_item['depth'] = self.to_pil(self.depth)

        if 'pseudo' in data_item:
            if not self._is_pil_image(self.pseudo):
                data_item['pseudo'] = self.to_pil(self.pseudo)
        if 'depth_interp' in data_item:
            if not self._is_pil_image(self.depth_interp):
                data_item['depth_interp'] = self.to_pil(self.depth_interp)

        return data_item


class ToTensor(BaseMethod):
    def __init__(self, mode="pair"):
        BaseMethod.__init__(self, mode=mode)
        self.totensor = transforms.ToTensor()

    def __call__(self, data_item):
        self.set_data(data_item)

        if self.mode in ["pair", "Img"]:
            data_item['img'] = self.totensor(self.img)
        if self.mode in ["pair", "depth"]:
            data_item['depth'] = self.totensor(self.depth).squeeze()
            if 'pseudo' in data_item:
                data_item['pseudo'] = self.totensor(self.pseudo).squeeze()
            if 'depth_interp' in data_item:
                data_item['depth_interp'] = self.totensor(self.depth_interp).squeeze()

        return data_item


class Crop(BaseMethod):
    def __init__(self, top, left, height, width):
        BaseMethod.__init__(self)
        self.height = height
        self.crop_pil_func = lambda pil: F.crop(pil, top, left, height, width)

    def __call__(self, data_item):
        self.set_data(data_item)

        data_item['img'] = self.crop_pil_func(self.img)
        if self.depth.size[1] > self.height:
            data_item['depth'] = self.crop_pil_func(self.depth)

        if 'depth_interp' in data_item:
            data_item['depth_interp'] = self.crop_pil_func(self.depth_interp)

        return data_item


class RandomCrop(BaseMethod):
    def __init__(self, height, width):
        BaseMethod.__init__(self)
        self.height = height
        self.width = width

    def __call__(self, data_item):
        self.set_data(data_item)

        top, left, height, width = transforms.RandomCrop.get_params(self.img, output_size=(self.height, self.width))
        randomcrop_func = lambda pil: F.crop(pil, top, left, height, width)

        data_item['img'] = randomcrop_func(self.img)
        data_item['depth'] = randomcrop_func(self.depth)
        if 'pseudo' in data_item:
            data_item['pseudo'] = randomcrop_func(self.pseudo)
        if 'depth_interp' in data_item:
            data_item['depth_interp'] = randomcrop_func(self.depth_interp)

        return data_item


class Img_Resize_Bilinear(BaseMethod):
    def __init__(self, kernel_size):
        BaseMethod.__init__(self)
        size = (int(375/kernel_size), int(1242/kernel_size))
        self.scale_rgb = transforms.Resize(size, Image.BILINEAR)

    def __call__(self, data_item):
        self.set_data(data_item)

        data_item['img'] = self.scale_rgb(self.img)

        return data_item


class Depth_Resize_MaxPool(BaseMethod):
    def __init__(self, kernel_size):
        BaseMethod.__init__(self)
        self.scale = nn.MaxPool2d(kernel_size=kernel_size)

    def __call__(self, data_item):
        self.set_data(data_item)

        data_item['depth'] = self.scale(self.depth[None, ...]).squeeze()
        if 'pseudo' in data_item:
            data_item['pseudo'] = self.scale(self.pseudo[None, ...]).squeeze()
        if 'depth_interp' in data_item:
            data_item['depth_interp'] = self.scale(self.depth_interp[None, ...]).squeeze()

        return data_item


class RandomHorizontalFlip(BaseMethod):
    def __init__(self):
        BaseMethod.__init__(self)

    def __call__(self, data_item):
        self.set_data(data_item)

        if random.random() < 0.5:
            data_item['img'] = self.img.transpose(Image.FLIP_LEFT_RIGHT)
            data_item['depth'] = self.depth.transpose(Image.FLIP_LEFT_RIGHT)
            if 'pseudo' in data_item:
                data_item['pseudo'] = self.pseudo.transpose(Image.FLIP_LEFT_RIGHT)
            if 'depth_interp' in data_item:
                data_item['depth_interp'] = self.depth_interp.transpose(Image.FLIP_LEFT_RIGHT)

        return data_item


class RandomRotate(BaseMethod):
    def __init__(self):
        BaseMethod.__init__(self)

    @staticmethod
    def rotate_pil_func():
        degree = random.randrange(-500, 500) / 100
        return lambda pil, interp: F.rotate(pil, degree, interp)

    def __call__(self, data_item):
        self.set_data(data_item)

        if random.random() < 0.5:
            rotate_pil = self.rotate_pil_func()
            data_item['img'] = rotate_pil(self.img, Image.BICUBIC)
            data_item['depth'] = rotate_pil(self.depth, Image.BILINEAR)
            if 'pseudo' in data_item:
                data_item['pseudo'] = rotate_pil(self.pseudo, Image.BILINEAR)
            if 'depth_interp' in data_item:
                data_item['depth_interp'] = rotate_pil(self.depth_interp, Image.BILINEAR)

        return data_item


class Img_Adjust(BaseMethod):
    def __init__(self):
        BaseMethod.__init__(self)

    @staticmethod
    def adjust_pil(pil):
        brightness = random.uniform(0.8, 1.0)
        contrast = random.uniform(0.8, 1.0)
        saturation = random.uniform(0.8, 1.0)

        pil = F.adjust_brightness(pil, brightness)
        pil = F.adjust_contrast(pil, contrast)
        pil = F.adjust_saturation(pil, saturation)

        return pil

    def __call__(self, data_item):
        self.set_data(data_item)
        data_item['img'] = self.adjust_pil(self.img)

        return data_item


class Img_Normalize(BaseMethod):
    def __init__(self, mean, std):
        BaseMethod.__init__(self)
        self.normalize = transforms.Normalize(mean, std)

    def __call__(self, data_item):
        self.set_data(data_item)
        data_item['img'] = self.normalize(self.img)

        return data_item
