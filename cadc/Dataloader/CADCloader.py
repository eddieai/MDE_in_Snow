#!usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pandas as pd
from PIL import Image


class CADCloader(object):
    """
    param kittiDir: KITTI dataset root path, e.g. ~/data/KITTI/
    param phase: 'train' or 'test'
    param cam: camera id. 2 represents the left cam, 3 represents the right one
    """
    def __init__(self, img_dir, depth_dir, phase, cam=0, snow_level=None, cam0cover=None, road_cover=None, depth_mode='aggregated', rescaled=False):
        self.img_root = img_dir if not rescaled else img_dir.replace('datasets', 'datasets_mod')
        self.depth_root = depth_dir
        self.phase = phase
        self.cam = cam
        self.snow_level = snow_level
        self.cam0cover = cam0cover
        self.road_cover = road_cover
        assert depth_mode in ['raw', 'dror', 'aggregated', 'HPR_3DBox', 'HPR_ConvexHull', 'HPR_ProjectedKNN', 'HPR_ProjectedKNN_0.99', 'aggregated_3', 'pseudo_dror', 'pseudo', 'pseudo_Kitti', 'dror_ProjectedKNN']
        self.depth_mode = depth_mode
        self.rescaled = rescaled
        self.files = []

        # read filenames files
        currpath = os.path.dirname(os.path.realpath(__file__))
        filepath = currpath + '/filenames/{}_files.txt'.format(self.phase)
        # load cadc_stats.csv for generation of different masks
        cadc_stats_path = os.path.dirname(currpath) + '/cadc_dataset_route_stats.csv'
        cadc_stats = pd.read_csv(cadc_stats_path, header=0, usecols=[0, 1, 2, 18, 19, 20, 21])
        cadc_filter = cadc_stats.copy()
        if self.snow_level is not None:
            assert self.snow_level in ['Light', 'Medium', 'Heavy', 'Extreme']
            cadc_filter = cadc_filter[cadc_filter.iloc[:, -1] == self.snow_level]
        if self.cam0cover is not None:
            assert isinstance(self.cam0cover, bool)
            if self.cam0cover:
                cadc_filter = cadc_filter[cadc_filter.iloc[:, -2] == 'Partial']
            else:
                cadc_filter = cadc_filter[cadc_filter.iloc[:, -2] == 'None']
        if self.road_cover is not None:
            assert isinstance(self.road_cover, bool)
            if self.road_cover:
                cadc_filter = cadc_filter[cadc_filter.iloc[:, -3] == 'Covered']
            else:
                cadc_filter = cadc_filter[cadc_filter.iloc[:, -3] == 'None']
        filtered_date = cadc_filter.iloc[:,0].values
        filtered_seq = list(map(lambda x: format(x, '04'), cadc_filter.iloc[:, 1].values))
        filtered = list(map('/'.join, list(zip(filtered_date, filtered_seq))))

        # if self.snow_level is not None:
        #     filepath = currpath + '/filenames/{}_files_{}.txt'.format(self.phase, self.snow_level)
        # if self.cam0cover is not None:
        #     if self.cam0cover:
        #         filepath = currpath + '/filenames/{}_files_{}.txt'.format(self.phase, 'cam0Cover')
        #     else:
        #         filepath = currpath + '/filenames/{}_files_{}.txt'.format(self.phase, 'cam0NoCover')
        # if self.road_cover is not None:
        #     if self.road_cover:
        #         filepath = currpath + '/filenames/{}_files_{}.txt'.format(self.phase, 'roadCover')
        #     else:
        #         filepath = currpath + '/filenames/{}_files_{}.txt'.format(self.phase, 'roadNoCover')

        with open(filepath, 'r') as f:
            data_list = f.read().split('\n')
            for data in data_list:
                if len(data) == 0:
                    continue
                if data[:15] not in filtered:
                    continue
                # remove first 5 and last 5 frames of each driving sequence
                # if self.depth_mode in ['aggregated', 'HPR_3DBox', 'HPR_ConvexHull', 'HPR_ProjectedKNN']:
                #     date = data[:10]
                #     seq = int(data[11:15])
                #     frame = int(data[-7:-5])
                #     frame_count = cadc_stats[(cadc_stats.Date == date) &
                #                              (cadc_stats.Number == seq)].iloc[:,2].values
                #     if not (5 <= frame < frame_count - 5):
                #         continue

                data_info = data.split(' ')

                if self.cam is not None:
                    if isinstance(self.cam, int):
                        cam_list = [self.cam]
                    elif isinstance(self.cam, list):
                        cam_list = self.cam
                else:
                    cam_list = list(range(8))

                for i in cam_list:
                    img_file = data_info[i]
                    if self.depth_mode == 'pseudo_dror':
                        depth_dror = data_info[i + 8].replace('depth', 'depth_dror')
                        pseudo_label = data_info[i + 8].replace('depth', 'pseudo_label')
                    elif self.depth_mode == 'pseudo':
                        depth_file = data_info[i + 8].replace('depth', 'pseudo_label')
                    elif self.depth_mode == 'pseudo_Kitti':
                        depth_file = data_info[i + 8].replace('depth', 'pseudo_label_pretrained_Kitti')
                    elif self.depth_mode == 'raw':
                        depth_file = data_info[i+8]
                    else:
                        depth_file = data_info[i+8].replace('depth', 'depth_%s' % self.depth_mode)

                    if self.depth_mode != 'pseudo_dror':
                        self.files.append({
                            "rgb": img_file if not self.rescaled else img_file.replace('labeled', 'rescaled'),
                            "depth": depth_file if not self.rescaled else depth_file.replace('labeled', 'rescaled'),
                        })
                    else:
                        self.files.append({
                            "rgb": img_file if not self.rescaled else img_file.replace('labeled', 'rescaled'),
                            "depth_dror": depth_dror if not self.rescaled else depth_dror.replace('labeled', 'rescaled'),
                            "pseudo_label": pseudo_label if not self.rescaled else pseudo_label.replace('labeled', 'rescaled'),
                        })

    def data_length(self):
        return len(self.files)

    def _check_path(self, item_files):
        if self.depth_mode != 'pseudo_dror':
            img_path = os.path.join(self.img_root, item_files['rgb'])
            depth_path = os.path.join(self.depth_root, item_files['depth'])
            assert os.path.exists(img_path), "Panic::Cannot find RGB image"
            assert os.path.exists(depth_path), "Panic::Cannot find depth map"
            return img_path, depth_path
        else:
            img_path = os.path.join(self.img_root, item_files['rgb'])
            depth_dror_path = os.path.join(self.depth_root, item_files['depth_dror'])
            pseudo_label_path = os.path.join(self.depth_root, item_files['pseudo_label'])
            assert os.path.exists(img_path), "Panic::Cannot find RGB image"
            assert os.path.exists(depth_dror_path), "Panic::Cannot find depth DROR"
            assert os.path.exists(pseudo_label_path), "Panic::Cannot find pseudo label"
            return img_path, depth_dror_path, pseudo_label_path

    def _read_depth(self, depth_path):
        # loads depth map D from png file
        # and returns it as a numpy array,89
        depth_png = np.array(Image.open(depth_path), dtype=int)
        # make sure we have a proper 16bit depth map here.. not 8bit!
        assert(np.max(depth_png) > 255)

        depth = depth_png.astype(np.float32) / 256.
        # depth[depth_png == 0] = -1.
        return depth

    def _read_data(self, item_files):
        if self.depth_mode != 'pseudo_dror':
            img_path, depth_path = self._check_path(item_files)
            img = Image.open(img_path).convert('RGB')
            depth = self._read_depth(depth_path)
            data = {'img': img, 'depth': depth}
        else:
            img_path, depth_dror_path, pseudo_label_path = self._check_path(item_files)
            img = Image.open(img_path).convert('RGB')
            depth = self._read_depth(depth_dror_path)
            pseudo = self._read_depth(pseudo_label_path)
            data = {'img': img, 'depth': depth, 'pseudo': pseudo}
        return data

    def load_item(self, idx):
        """
        load an item for training or test
        interp_method can be selected from ['nop', 'linear', 'nyu']
        """
        item_files = self.files[idx]
        data_item = self._read_data(item_files)
        return data_item
