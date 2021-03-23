from tqdm import *
from PIL import Image
import os
import numpy as np

rescale_types = ['depth_inference_pretrained_Kitti', 'kl_div_output_pred_pretrained_Kitti', 'pseudo_label_pretrained_Kitti']
rescaled_size = (640, 257)
show_rescaled = False

all_files_txt = '../all_seq_files.txt'
BASE = '/home/datasets/CADC/cadcd/'
BASE_mod = '/home/datasets_mod/CADC/cadcd/'
if show_rescaled:
    import matplotlib.pyplot as plt
    plt.ion()

with open(all_files_txt, 'r') as all_files:
    data = all_files.read().split('\n')

for row in tqdm(data):
    labeled_path_template = row.split(' ')[0]
    labeled_paths = [labeled_path_template.replace('data', rescale_type) for rescale_type in rescale_types]

    for labeled_path in labeled_paths:
        if 'data' in labeled_path:
            img = Image.open(BASE + labeled_path)
            rescaled = img.resize(rescaled_size, resample=Image.BILINEAR)
            if show_rescaled:
                plt.imshow(rescaled)
        else:
            depth = Image.open(BASE_mod + labeled_path)
            rescaled = depth.resize(rescaled_size, resample=Image.NEAREST)
            if show_rescaled:
                plt.imshow(np.array(rescaled, dtype=int).astype(np.float32) / 256., cmap='jet')

        rescaled_path = BASE_mod + labeled_path.replace('labeled', 'rescaled')
        if not (os.path.exists(rescaled_path[:-14])):
            os.makedirs(rescaled_path[:-14])
        rescaled.save(rescaled_path)

        if show_rescaled:
            plt.pause(0.001)
            plt.show()
