import os
import torch
import numpy as np
from PIL import Image
from .Dataloader import CADCloader
from .Transformer import Transformer
from torch.utils.data import Dataset, DataLoader


class CADCDataset(Dataset):
    def __init__(self,
                 img_dir,
                 depth_dir,
                 phase,
                 cam,
                 snow_level,
                 cam0cover,
                 road_cover,
                 depth_mode,
                 rescaled,
                 transform):
        self.transform = transform

        # use left image by default
        self.cadcloader = CADCloader(img_dir,
                                     depth_dir,
                                     phase,
                                     cam,
                                     snow_level,
                                     cam0cover,
                                     road_cover,
                                     depth_mode,
                                     rescaled)

    def __getitem__(self, idx):
        # load an item according to the given index
        data_item = self.cadcloader.load_item(idx)
        data_transed = self.transform(data_item)
        return data_transed

    def __len__(self):
        return self.cadcloader.data_length()


class DataGenerator(object):
    def __init__(self,
                 img_dir,
                 depth_dir,
                 phase,
                 cam=0,
                 snow_level=None,
                 cam0cover=None,
                 road_cover=None,
                 depth_mode='aggregated',
                 rescaled=False,
                 high_gpu=True):
        self.phase = phase
        self.rescaled = rescaled
        self.depth_mode = depth_mode
        self.high_gpu = high_gpu

        if not self.phase in ['train', 'test', 'val', 'small', 'all', 'inference', 'train_seq', 'val_seq', 'test_seq', 'all_seq']:
            raise ValueError("Panic::Invalid phase parameter")
        else:
            pass

        transformer = Transformer(self.phase, self.rescaled)
        self.dataset = CADCDataset(img_dir,
                                   depth_dir,
                                   phase,
                                   cam,
                                   snow_level,
                                   cam0cover,
                                   road_cover,
                                   depth_mode,
                                   rescaled,
                                   transformer.get_transform())

    def create_data(self, batch_size, nthreads=0):
        # use page locked gpu memory by default
        return DataLoader(self.dataset,
                          batch_size,
                          shuffle=(self.phase in ['train', 'small', 'train_seq']),
                          num_workers=nthreads,
                          pin_memory=self.high_gpu,
                          drop_last=True)
