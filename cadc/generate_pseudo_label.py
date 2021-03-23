from tqdm import *
import os
import numpy as np
import pandas as pd
from PIL import Image

BASE = "/home/datasets/CADC/cadcd/"
BASE_mod = "/home/datasets_mod/CADC/cadcd/"
cadc_stats = pd.read_csv('cadc_dataset_route_stats.csv', header=0, usecols=[0, 1, 2, 18, 19, 20, 21])
start_row = 21
start_frame = 98

threshold = 0.5

for row in trange(start_row, len(cadc_stats)): # len(cadc_stats)
    # print('\n------ Date %s, Sequence %d ------' % (cadc_stats.iloc[row, 0], cadc_stats.iloc[row, 1]))
    date = cadc_stats.iloc[row, 0]
    seq = cadc_stats.iloc[row, 1]
    n_frame = cadc_stats.iloc[row, 2]

    for frame in trange(start_frame, n_frame):
        try:
            kl_div_png_path = BASE_mod + date + '/' + format(seq, '04') + "/labeled/image_00/kl_div_output_pred_pretrained_Kitti/" + format(frame, '010') + ".png"
            kl_div_png = np.array(Image.open(kl_div_png_path), dtype=int)
            kl_div = kl_div_png.astype(np.float32) / 256.

            depth_inference_png_path = BASE_mod + date + '/' + format(seq, '04') + "/labeled/image_00/depth_inference_pretrained_Kitti/" + format(frame, '010') + ".png"
            depth_inference_png = np.array(Image.open(depth_inference_png_path), dtype=int)
            depth_inference = depth_inference_png.astype(np.float32) / 256.

            pseudo_label = np.where(kl_div < threshold, depth_inference, 0)

            pseudo_label_path = BASE_mod + date + '/' + format(seq, '04') + "/labeled/image_00/pseudo_label_pretrained_Kitti/" + format(frame, '010') + ".png"
            if not (os.path.exists(pseudo_label_path[:-14])):
                os.makedirs(pseudo_label_path[:-14])
            pseudo_label_PIL = Image.fromarray(np.clip(pseudo_label * 256., 0, 65535)).convert('I')
            pseudo_label_PIL.save(pseudo_label_path, mode='I;16')

        except:
            continue
