import torch
import numpy as np
import torchvision.transforms as transforms
from cadc.Transformer import custom_methods as augmethods
from .base_transformer import BaseTransformer


class CustTransformer(BaseTransformer):
    """
    An example of Custom Transformer.
    This class should work with custom transform methods which defined in custom_methods.py
    """

    def __init__(self, phase, rescaled):
        BaseTransformer.__init__(self, phase, rescaled)
        if not self.phase in ['train', 'test', 'val', 'small', 'all', 'inference', 'train_seq', 'val_seq', 'test_seq', 'all_seq']:
            raise ValueError("Panic::Invalid phase parameter")
        else:
            pass

    def get_joint_transform(self):
        if not self.rescaled:
            if self.phase in ["train", "small", "train_seq"]:
                return transforms.Compose([
                    augmethods.ToPIL(),
                    augmethods.Crop(200, 0, 513, 1280),   # (250, 0, 513, 1280)  (350, 0, 375, 1280)
                    augmethods.RandomCrop(513, 513),
                    augmethods.RandomHorizontalFlip(),
                    # augmethods.RandomRotate()
                ])
            else:
                return transforms.Compose([
                    augmethods.ToPIL(),
                    augmethods.Crop(200, 0, 513, 1280)
                    # augmethods.RandomCrop(375, 513)
                ])
        else:
            if self.phase in ["train", "small", "train_seq"]:
                return transforms.Compose([
                    augmethods.ToPIL(),
                    augmethods.Crop(100, 0, 257, 640),   # (250, 0, 513, 1280)  (350, 0, 375, 1280)
                    augmethods.RandomCrop(257, 257),
                    augmethods.RandomHorizontalFlip(),
                    # augmethods.RandomRotate()
                ])
            else:
                return transforms.Compose([
                    augmethods.ToPIL(),
                    augmethods.Crop(100, 0, 257, 640)
                    # augmethods.RandomCrop(375, 513)
                ])

    def get_img_transform(self):
        if self.phase in ["train", "small", "train_seq"]:
            return transforms.Compose([
                # augmethods.Img_Adjust(),
                # augmethods.Img_Resize_Bilinear(kernel_size=3),
                augmethods.ToTensor("Img"),
                # augmethods.Img_Normalize([.5, .5, .5], [.5, .5, .5])
            ])
        else:
            return transforms.Compose([
                # augmethods.Img_Resize_Bilinear(kernel_size=3),
                augmethods.ToTensor("Img"),
                # augmethods.Img_Normalize([.5, .5, .5], [.5, .5, .5])
            ])

    def get_depth_transform(self):
        if self.phase in ["train", "small", "train_seq"]:
            return transforms.Compose([
                augmethods.ToTensor("depth"),
                # augmethods.Depth_Resize_MaxPool(kernel_size=3)
            ])
        else:
            return transforms.Compose([
                augmethods.ToTensor("depth"),
                # augmethods.Depth_Resize_MaxPool(kernel_size=3)
            ])
