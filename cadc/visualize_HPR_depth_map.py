from tqdm import *
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import cm

from cadc_utils import *
from filter_lidar_HPR_ConvexHull import HPR_ConvexHull
from filter_lidar_HPR_ProjectedKNN import HPR_ProjectedKNN


BASE = "/home/datasets/CADC/cadcd/"
BASE_mod = "/home/datasets_mod/CADC/cadcd/"
cadc_stats = pd.read_csv('cadc_dataset_route_stats.csv', header=0, usecols=[0, 1, 2, 18, 19, 20, 21])
start_row = 36   # 25
start_frame = 80
cam = 0

for row in trange(start_row, start_row+1):  # len(cadc_stats)
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

    # T_IMG_CAM = np.eye(4)
    # T_IMG_CAM[0:3, 0:3] = np.array(calib['CAM00']['camera_matrix']['data']).reshape(-1, 3)
    # T_IMG_CAM = T_IMG_CAM[0:3, 0:4]  # remove last row
    # T_CAM_LIDAR = np.linalg.inv(np.array(calib['extrinsics']['T_LIDAR_CAM00']))

    for frame in trange(start_frame, start_frame+10):       # n_frame
        # visualization
        lidar_aggregate_path = BASE_mod + date + '/' + format(seq, '04') + \
                                 "/labeled/lidar_points/lidar_aggregated/" + \
                                 format(frame, '010') + ".npy"
        lidar_aggregate = np.load(lidar_aggregate_path).reshape((-1, 3))
        # Project points onto cam0 image plane
        lidar_aggregate_projected = Lidar2Cam(lidar_aggregate, T_IMG_CAM, T_CAM_LIDAR)
        # Crop lidar point cloud to cam0 field of view
        x = lidar_aggregate_projected[:, 0]
        y = lidar_aggregate_projected[:, 1]
        depth = lidar_aggregate_projected[:, 2]
        lidar_aggregate_cropped = lidar_aggregate[np.logical_and.reduce((x >= 0, x < w, y >= 0, y < h, depth >= 0))]
        # Crop projected points to cam0 field of view
        lidar_aggregate_projected_cropped = CropPoints(lidar_aggregate_projected)

        # # Generate cam0 depth map (H, W) from points (N, 3)
        depth_aggregate = GenerateDepth(lidar_aggregate_projected_cropped)
        # depth map of aggregated lidar
        # depth_aggregate = depth_aggregate.clip(0, 80) / 80
        # im_depth_aggregate = Image.fromarray(np.uint8(cm.jet(depth_aggregate) * 255))
        # im_depth_aggregate.save('save_image/depth_aggregate_%d_corrected.png' % frame)

        # depth map of aggregated lidar after HPR ConvexHull
        # filtered_HPR_ConvexHull = HPR_ConvexHull(lidar_aggregate_cropped, param=2)
        # depth_HPR_ConvexHull = GenerateDepth(lidar_aggregate_projected_cropped[filtered_HPR_ConvexHull])
        # depth_HPR_ConvexHull = depth_HPR_ConvexHull.clip(0, 80) / 80
        # im_depth_HPR_ConvexHull = Image.fromarray(np.uint8(cm.jet(depth_HPR_ConvexHull) * 255))
        # im_depth_HPR_ConvexHull.save('save_image/depth_HPR_ConvexHull_%d_corrected.png' % frame)

        # depth map of aggregated lidar after HPR ProjectedKNN
        filtered_HPR_ProjectedKNN = HPR_ProjectedKNN(lidar_aggregate_cropped, lidar_aggregate_projected_cropped[:, :2], (h, w), K=225, alpha=0.99)
        depth_HPR_ProjectedKNN = GenerateDepth(lidar_aggregate_projected_cropped[filtered_HPR_ProjectedKNN])
        depth_HPR_ProjectedKNN = depth_HPR_ProjectedKNN.clip(0, 80) / 80
        im_depth_HPR_ProjectedKNN = Image.fromarray(np.uint8(cm.jet(depth_HPR_ProjectedKNN) * 255))
        im_depth_HPR_ProjectedKNN.save('save_image/depth_HPR_ProjectedKNN_%d_corrected.png' % frame)
