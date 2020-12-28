import os
import torch
import numpy as np
from PIL import Image
from .Dataloader import Kittiloader
from .Transformer import Transformer
from torch.utils.data import Dataset, DataLoader


class KittiDataset(Dataset):
    def __init__(self,
                 kittiDir,
                 mode,
                 interp_method,
                 transform):
        # self.mode = mode
        # self.kitti_root = kittiDir
        self.transform = transform

        # use left image by default
        self.kittiloader = Kittiloader(kittiDir, mode, interp_method=interp_method, cam=2)

    def __getitem__(self, idx):
        # load an item according to the given index
        data_item = self.kittiloader.load_item(idx)
        data_transed = self.transform(data_item)
        return data_transed

    def __len__(self):
        return self.kittiloader.data_length()


class DataGenerator(object):
    def __init__(self,
                 KittiDir,
                 phase,
                 interp_method='nop',
                 high_gpu=True, ):
        self.phase = phase
        self.high_gpu = high_gpu

        if not self.phase in ['train', 'test', 'val', 'small']:
            raise ValueError("Panic::Invalid phase parameter")
        else:
            pass

        transformer = Transformer(self.phase)
        self.dataset = KittiDataset(KittiDir,
                                    phase,
                                    interp_method,
                                    transformer.get_transform())

    def create_data(self, batch_size, nthreads=0):
        # use page locked gpu memory by default
        return DataLoader(self.dataset,
                          batch_size,
                          shuffle=(self.phase == 'train' or self.phase == 'small'),
                          num_workers=nthreads,
                          pin_memory=self.high_gpu,
                          drop_last=True)
