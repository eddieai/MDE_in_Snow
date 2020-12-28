from tqdm import *
import numpy as np
import pandas as pd
import PIL.Image as Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from cadc_utils import Lidar2Cam, GenerateDepth, load_calibration, CropPoints

plt.ion()
_, ax = plt.subplots(2, 3)
cmap = cm.get_cmap('jet')

BASE = "/home/datasets/CADC/cadcd/"
BASE_mod = "/home/datasets_mod/CADC/cadcd/"
cadc_stats = pd.read_csv('cadc_dataset_route_stats.csv', header=0, usecols=[0, 1, 2, 18, 19, 20, 21])
start_row = 32   # 25
start_frame = 20
cam = 0

for row in trange(start_row, len(cadc_stats)):
    print('\n------ Date %s, Sequence %d ------' % (cadc_stats.iloc[row, 0], cadc_stats.iloc[row, 1]))
    date = cadc_stats.iloc[row, 0]
    seq = cadc_stats.iloc[row, 1]
    n_frame = cadc_stats.iloc[row, 2]

    calib_path = BASE + date + '/' + "calib/"
    calib = load_calibration(calib_path)
    # Projection matrix from lidar to camera (extrinsics)
    T_CAM_LIDAR = np.linalg.inv(np.array(calib['extrinsics']['T_LIDAR_CAM0' + str(cam)]))[:3, :4]
    # Projection matrix from camera to image (intrinsics)
    T_IMG_CAM = np.array(calib['CAM0' + str(cam)]['camera_matrix']['data']).reshape(3, 3)

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
        # lidar_aggregate_path = BASE_mod + date + '/' + format(seq, '04') + \
        #                          "/labeled/lidar_points/lidar_aggregated/" + \
        #                          format(frame, '010') + ".npy"
        # lidar_aggregate = np.load(lidar_aggregate_path).reshape((-1, 3))
        # # Project points onto image
        # lidar_aggregate_projected = Lidar2Cam(lidar_aggregate, T_IMG_CAM, T_CAM_LIDAR)
        # # Crop points to image view field
        # lidar_aggregate_cropped = CropPoints(lidar_aggregate_projected)
        # # Generate depth map (H, W) from points (N, 3)
        # depth_aggregate = GenerateDepth(lidar_aggregate_cropped)
        depth_agg_culling_path = BASE_mod + date + '/' + format(seq, '04') + "/labeled" + "/image_0" + str(cam) + \
                   "/depth_aggregated_5/" + format(frame, '010') + ".png"
        depth_agg_culling_5 = np.array(Image.open(depth_agg_culling_path), dtype=int).astype(np.float32) / 256.
        depth_agg_culling_path = BASE_mod + date + '/' + format(seq, '04') + "/labeled" + "/image_0" + str(cam) + \
                   "/depth_aggregated/" + format(frame, '010') + ".png"
        depth_agg_culling_11 = np.array(Image.open(depth_agg_culling_path), dtype=int).astype(np.float32) / 256.
        depth_agg_culling_path = BASE_mod + date + '/' + format(seq, '04') + "/labeled" + "/image_0" + str(cam) + \
                   "/depth_aggregated_21/" + format(frame, '010') + ".png"
        depth_agg_culling_21 = np.array(Image.open(depth_agg_culling_path), dtype=int).astype(np.float32) / 256.

        ax[0,0].clear()
        ax[0,0].imshow(img)
        ax[0,0].set_title('RGB image')
        ax[0, 0].get_xaxis().set_visible(False)
        ax[0,0].get_yaxis().set_visible(False)
        ax[0,1].clear()
        ax[0,1].imshow(img)
        ax[0,1].scatter(np.nonzero(depth_raw)[1], np.nonzero(depth_raw)[0],
                        c=cmap(np.clip(depth_raw[np.nonzero(depth_raw)], 0, 80) / 80), s=0.0005)
        ax[0,1].set_title('Raw Depth map')
        ax[0, 1].get_xaxis().set_visible(False)
        ax[0,1].get_yaxis().set_visible(False)
        ax[0,2].clear()
        ax[0,2].imshow(img)
        ax[0,2].scatter(np.nonzero(depth_dror)[1], np.nonzero(depth_dror)[0],
                        c=cmap(np.clip(depth_dror[np.nonzero(depth_dror)], 0, 80) / 80), s=0.0005)
        ax[0,2].set_title('DROR Depth map')
        ax[0, 2].get_xaxis().set_visible(False)
        ax[0,2].get_yaxis().set_visible(False)
        # ax[3].clear()
        # ax[3].imshow(img)
        # ax[3].scatter(np.nonzero(depth_aggregate)[1], np.nonzero(depth_aggregate)[0],
        #                 c=cmap(np.clip(depth_aggregate[np.nonzero(depth_aggregate)], 0, 80) / 80), s=0.01)
        # ax[3].set_title('Aggregated Depth map (no culling)')
        ax[1,0].clear()
        ax[1,0].imshow(img)
        ax[1,0].scatter(np.nonzero(depth_agg_culling_5)[1], np.nonzero(depth_agg_culling_5)[0],
                        c=cmap(np.clip(depth_agg_culling_5[np.nonzero(depth_agg_culling_5)], 0, 80) / 80), s=0.0005)
        ax[1,0].set_title('Aggregated Depth map (5 frames)')
        ax[1, 0].get_xaxis().set_visible(False)
        ax[1,0].get_yaxis().set_visible(False)
        ax[1,1].clear()
        ax[1,1].imshow(img)
        ax[1,1].scatter(np.nonzero(depth_agg_culling_11)[1], np.nonzero(depth_agg_culling_11)[0],
                        c=cmap(np.clip(depth_agg_culling_11[np.nonzero(depth_agg_culling_11)], 0, 80) / 80), s=0.0005)
        ax[1,1].set_title('Aggregated Depth map (11 frames)')
        ax[1, 1].get_xaxis().set_visible(False)
        ax[1,1].get_yaxis().set_visible(False)
        ax[1,2].clear()
        ax[1,2].imshow(img)
        ax[1,2].scatter(np.nonzero(depth_agg_culling_21)[1], np.nonzero(depth_agg_culling_21)[0],
                        c=cmap(np.clip(depth_agg_culling_21[np.nonzero(depth_agg_culling_21)], 0, 80) / 80), s=0.0005)
        ax[1,2].set_title('Aggregated Depth map (21 frames)')
        ax[1, 2].get_xaxis().set_visible(False)
        ax[1,2].get_yaxis().set_visible(False)

        plt.pause(0.1)
        plt.show()
