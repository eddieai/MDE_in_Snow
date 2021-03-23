from tqdm import *
import numpy as np
import pandas as pd
import PIL.Image as Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
# plt.ion()
fig, ax = plt.subplots(3, 3)
fig.tight_layout()

BASE = "/home/datasets/CADC/cadcd/"
BASE_mod = "/home/datasets_mod/CADC/cadcd/"
cadc_stats = pd.read_csv('cadc_dataset_route_stats.csv', header=0, usecols=[0, 1, 2, 18, 19, 20, 21])
start_row = 9   # 25
start_frame = 0
RANDOM = True
n_png = 100
cam = 0

if RANDOM:
    row_range = tqdm(np.random.randint(0, len(cadc_stats), size=n_png))
else:
    row_range = trange(start_row, len(cadc_stats))  # len(cadc_stats)

n = 0
for row in row_range:
    print('\n------ Date %s, Sequence %d ------' % (cadc_stats.iloc[row, 0], cadc_stats.iloc[row, 1]))
    date = cadc_stats.iloc[row, 0]
    seq = cadc_stats.iloc[row, 1]
    n_frame = cadc_stats.iloc[row, 2]

    if RANDOM:
        frame_range = [np.random.randint(0, n_frame)]
    else:
        frame_range = trange(start_frame, n_frame)     # n_frame

    for frame in frame_range:
        img_path = BASE + date + '/' + format(seq, '04') + "/labeled" + "/image_0" + str(cam) + \
                   "/data/" + format(frame, '010') + ".png"
        img = plt.imread(img_path)

        depth_dror_path = BASE_mod + date + '/' + format(seq, '04') + "/labeled" + "/image_0" + str(cam) + \
                          "/depth_dror/" + format(frame, '010') + ".png"
        depth_dror = np.array(Image.open(depth_dror_path), dtype=int).astype(np.float32) / 256.

        depth_HPR_ProjectedKNN_path = BASE_mod + date + '/' + format(seq, '04') + "/labeled" + "/image_0" + str(cam) + \
                                      "/depth_HPR_ProjectedKNN/" + format(frame, '010') + ".png"
        depth_HPR_ProjectedKNN = np.array(Image.open(depth_HPR_ProjectedKNN_path), dtype=int).astype(np.float32) / 256.

        depth_inference_path = BASE_mod + date + '/' + format(seq, '04') + "/labeled" + "/image_0" + str(cam) + \
                                      "/depth_inference_pretrained_Kitti/" + format(frame, '010') + ".png"
        depth_inference = np.array(Image.open(depth_inference_path), dtype=int).astype(np.float32) / 256.

        kl_div_output_pred_path = BASE_mod + date + '/' + format(seq, '04') + "/labeled" + "/image_0" + str(cam) + \
                                      "/kl_div_output_pred_pretrained_Kitti/" + format(frame, '010') + ".png"
        kl_div_output_pred = np.array(Image.open(kl_div_output_pred_path), dtype=int).astype(np.float32) / 256.

        pseudo_label_path = BASE_mod + date + '/' + format(seq, '04') + "/labeled" + "/image_0" + str(cam) + \
                                      "/pseudo_label_pretrained_Kitti/" + format(frame, '010') + ".png"
        pseudo_label = np.array(Image.open(pseudo_label_path), dtype=int).astype(np.float32) / 256.

        pseudo_label_0_1 = np.where(kl_div_output_pred < 0.1, depth_inference, 0)
        pseudo_label_0_3 = np.where(kl_div_output_pred < 0.3, depth_inference, 0)
        pseudo_label_0_8 = np.where(kl_div_output_pred < 0.8, depth_inference, 0)

        ax[0, 0].clear()
        ax[0, 0].imshow(img[200:713, :, :])
        ax[0, 0].set_title('RGB image', fontsize=6)
        ax[0, 0].get_xaxis().set_visible(False)
        ax[0, 0].get_yaxis().set_visible(False)
        ax[0, 1].clear()
        ax[0, 1].imshow(img[200:713, :, :])
        ax[0, 1].imshow(np.ma.masked_where(depth_dror[200:713, :] == 0, depth_dror[200:713, :].clip(0, 80)), cmap='jet', vmin=0, vmax=80,
                        alpha=0.5)
        ax[0, 1].set_title('Depth map after DROR', fontsize=6)
        ax[0, 1].get_xaxis().set_visible(False)
        ax[0, 1].get_yaxis().set_visible(False)
        ax[0, 2].clear()
        ax[0, 2].imshow(img[200:713, :, :])
        ax[0, 2].imshow(np.ma.masked_where(depth_HPR_ProjectedKNN[200:713, :] == 0, depth_HPR_ProjectedKNN[200:713, :].clip(0, 80)), cmap='jet', vmin=0, vmax=80,
                        alpha=0.5)
        ax[0, 2].set_title('Aggregate 11 frames + ProjectedKNN', fontsize=6)
        ax[0, 2].get_xaxis().set_visible(False)
        ax[0, 2].get_yaxis().set_visible(False)

        ax[1, 0].clear()
        ax[1, 0].imshow(img[200:713, :, :])
        ax[1, 0].imshow(depth_inference.clip(0, 80), cmap='jet', vmin=0, vmax=80, alpha=0.5)
        ax[1, 0].set_title('Depth prediction', fontsize=6)
        ax[1, 0].get_xaxis().set_visible(False)
        ax[1, 0].get_yaxis().set_visible(False)
        ax[1, 1].clear()
        ax[1, 1].imshow(img[200:713, :, :])
        ax[1, 1].imshow(kl_div_output_pred.clip(0, 1), cmap='viridis', vmin=0, vmax=1, alpha=0.5)
        ax[1, 1].set_title('KL-div (Output/Prediction)', fontsize=6)
        ax[1, 1].get_xaxis().set_visible(False)
        ax[1, 1].get_yaxis().set_visible(False)
        ax[1, 2].clear()
        ax[1, 2].imshow(img[200:713, :, :])
        ax[1, 2].imshow(np.ma.masked_where(pseudo_label == 0, pseudo_label.clip(0, 80)), cmap='jet', vmin=0,
                        vmax=80, alpha=0.5)
        ax[1, 2].set_title('Pseudo Label < threshold 0.5', fontsize=6)
        ax[1, 2].get_xaxis().set_visible(False)
        ax[1, 2].get_yaxis().set_visible(False)

        ax[2, 0].clear()
        ax[2, 0].imshow(img[200:713, :, :])
        ax[2, 0].imshow(np.ma.masked_where(pseudo_label_0_1 == 0, pseudo_label_0_1.clip(0, 80)), cmap='jet', vmin=0,
                        vmax=80, alpha=0.5)
        ax[2, 0].set_title('Pseudo Label < threshold 0.1', fontsize=6)
        ax[2, 0].get_xaxis().set_visible(False)
        ax[2, 0].get_yaxis().set_visible(False)
        ax[2, 1].clear()
        ax[2, 1].imshow(img[200:713, :, :])
        ax[2, 1].imshow(np.ma.masked_where(pseudo_label_0_3 == 0, pseudo_label_0_3.clip(0, 80)), cmap='jet', vmin=0,
                        vmax=80, alpha=0.5)
        ax[2, 1].set_title('Pseudo Label < threshold 0.3', fontsize=6)
        ax[2, 1].get_xaxis().set_visible(False)
        ax[2, 1].get_yaxis().set_visible(False)
        ax[2, 2].clear()
        ax[2, 2].imshow(img[200:713, :, :])
        ax[2, 2].imshow(np.ma.masked_where(pseudo_label_0_8 == 0, pseudo_label_0_8.clip(0, 80)), cmap='jet', vmin=0,
                        vmax=80, alpha=0.5)
        ax[2, 2].set_title('Pseudo Label < threshold 0.8', fontsize=6)
        ax[2, 2].get_xaxis().set_visible(False)
        ax[2, 2].get_yaxis().set_visible(False)

        # plt.pause(0.1)
        # plt.show()
        plt.savefig('visualize_pseudo_label_pretrained_Kitti/Date_%s_Seq_%2d_Frame_%2d.png' % (date, seq, frame), dpi=600)

        n += 1
        if n >= n_png:
            break
