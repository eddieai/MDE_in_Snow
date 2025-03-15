from tqdm import *
import numpy as np
import pandas as pd
import PIL.Image as Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

plt.ion()
_, ax = plt.subplots(2, 4)
cmap = cm.get_cmap('jet')

BASE = "/home/datasets/CADC/cadcd/"
BASE_mod = "/home/datasets_mod/CADC/cadcd/"
cadc_stats = pd.read_csv('cadc_dataset_route_stats.csv', header=0, usecols=[0, 1, 2, 18, 19, 20, 21])
start_row = 1   # 25
start_frame = 0
cam = 0

for row in trange(start_row, len(cadc_stats)):
    print('\n------ Date %s, Sequence %d ------' % (cadc_stats.iloc[row, 0], cadc_stats.iloc[row, 1]))
    date = cadc_stats.iloc[row, 0]
    seq = cadc_stats.iloc[row, 1]
    n_frame = cadc_stats.iloc[row, 2]

    for frame in trange(start_frame, n_frame):
        # visualization
        img_path = BASE + date + '/' + format(seq, '04') + "/labeled" + "/image_0" + str(cam) + \
                   "/data/" + format(frame, '010') + ".png"
        img = plt.imread(img_path)
        depth_raw_path = BASE_mod + date + '/' + format(seq, '04') + "/labeled" + "/image_0" + str(cam) + \
                         "/depth/" + format(frame, '010') + ".png"
        depth_raw = np.array(Image.open(depth_raw_path), dtype=int).astype(np.float32) / 256.
        depth_dror_path = BASE_mod + date + '/' + format(seq, '04') + "/labeled" + "/image_0" + str(cam) + \
                          "/depth_dror/" + format(frame, '010') + ".png"
        depth_dror = np.array(Image.open(depth_dror_path), dtype=int).astype(np.float32) / 256.
        depth_agg_culling_path = BASE_mod + date + '/' + format(seq, '04') + "/labeled" + "/image_0" + str(cam) + \
                                 "/depth_aggregated/" + format(frame, '010') + ".png"
        depth_agg_culling = np.array(Image.open(depth_agg_culling_path), dtype=int).astype(np.float32) / 256.

        img_rescaled_path = BASE_mod + date + '/' + format(seq, '04') + "/rescaled" + "/image_0" + str(cam) + \
                   "/data/" + format(frame, '010') + ".png"
        img_rescaled = plt.imread(img_rescaled_path)
        depth_raw_rescaled_path = BASE_mod + date + '/' + format(seq, '04') + "/rescaled" + "/image_0" + str(cam) + \
                         "/depth/" + format(frame, '010') + ".png"
        depth_raw_rescaled = np.array(Image.open(depth_raw_rescaled_path), dtype=int).astype(np.float32) / 256.
        depth_dror_rescaled_path = BASE_mod + date + '/' + format(seq, '04') + "/rescaled" + "/image_0" + str(cam) + \
                          "/depth_dror/" + format(frame, '010') + ".png"
        depth_dror_rescaled = np.array(Image.open(depth_dror_rescaled_path), dtype=int).astype(np.float32) / 256.
        depth_agg_culling_rescaled_path = BASE_mod + date + '/' + format(seq, '04') + "/rescaled" + "/image_0" + str(cam) + \
                                 "/depth_aggregated/" + format(frame, '010') + ".png"
        depth_agg_culling_rescaled = np.array(Image.open(depth_agg_culling_rescaled_path), dtype=int).astype(np.float32) / 256.

        ax[0,0].clear()
        ax[0,0].imshow(img)
        ax[0,0].set_title('RGB image')
        # ax[0,0].get_xaxis().set_visible(False)
        # ax[0,0].get_yaxis().set_visible(False)

        ax[0,1].clear()
        ax[0,1].imshow(img)
        ax[0,1].scatter(np.nonzero(depth_raw)[1], np.nonzero(depth_raw)[0],
                        c=cmap(np.clip(depth_raw[np.nonzero(depth_raw)], 0, 80) / 80), s=0.0025)
        ax[0,1].set_title('Raw Depth map')
        # ax[0,1].get_xaxis().set_visible(False)
        # ax[0,1].get_yaxis().set_visible(False)

        ax[0,2].clear()
        ax[0,2].imshow(img)
        ax[0,2].scatter(np.nonzero(depth_dror)[1], np.nonzero(depth_dror)[0],
                        c=cmap(np.clip(depth_dror[np.nonzero(depth_dror)], 0, 80) / 80), s=0.0025)
        ax[0,2].set_title('DROR Depth map')
        # ax[0,2].get_xaxis().set_visible(False)
        # ax[0,2].get_yaxis().set_visible(False)

        ax[0,3].clear()
        ax[0,3].imshow(img)
        ax[0,3].scatter(np.nonzero(depth_agg_culling)[1], np.nonzero(depth_agg_culling)[0],
                        c=cmap(np.clip(depth_agg_culling[np.nonzero(depth_agg_culling)], 0, 80) / 80), s=0.0025)
        ax[0,3].set_title('Aggregated Depth map (11 frames)')
        # ax[0,3].get_xaxis().set_visible(False)
        # ax[0,3].get_yaxis().set_visible(False)

        ax[1, 0].clear()
        ax[1, 0].imshow(img_rescaled)
        ax[1, 0].set_title('Rescaled RGB image')
        # ax[1, 0].get_xaxis().set_visible(False)
        # ax[1, 0].get_yaxis().set_visible(False)

        ax[1, 1].clear()
        ax[1, 1].imshow(img_rescaled)
        ax[1, 1].scatter(np.nonzero(depth_raw_rescaled)[1], np.nonzero(depth_raw_rescaled)[0],
                         c=cmap(np.clip(depth_raw_rescaled[np.nonzero(depth_raw_rescaled)], 0, 80) / 80), s=0.05)
        ax[1, 1].set_title('Rescaled Raw Depth map')
        # ax[1, 1].get_xaxis().set_visible(False)
        # ax[1, 1].get_yaxis().set_visible(False)

        ax[1, 2].clear()
        ax[1, 2].imshow(img_rescaled)
        ax[1, 2].scatter(np.nonzero(depth_dror_rescaled)[1], np.nonzero(depth_dror_rescaled)[0],
                         c=cmap(np.clip(depth_dror_rescaled[np.nonzero(depth_dror_rescaled)], 0, 80) / 80), s=0.05)
        ax[1, 2].set_title('Rescaled DROR Depth map')
        # ax[1, 2].get_xaxis().set_visible(False)
        # ax[1, 2].get_yaxis().set_visible(False)

        ax[1, 3].clear()
        ax[1, 3].imshow(img_rescaled)
        ax[1, 3].scatter(np.nonzero(depth_agg_culling_rescaled)[1], np.nonzero(depth_agg_culling_rescaled)[0],
                         c=cmap(np.clip(depth_agg_culling_rescaled[np.nonzero(depth_agg_culling_rescaled)], 0, 80) / 80), s=0.05)
        ax[1, 3].set_title('Rescaled Aggregated Depth map')
        # ax[1, 3].get_xaxis().set_visible(False)
        # ax[1, 3].get_yaxis().set_visible(False)

        plt.pause(0.1)
        plt.show()
