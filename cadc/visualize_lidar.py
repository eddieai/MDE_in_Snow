from tqdm import *
import numpy as np
import pandas as pd
import open3d as o3d
from cadc_utils import visualize_3d


BASE = "/home/datasets/CADC/cadcd/"
BASE_mod = "/home/datasets_mod/CADC/cadcd/"
cadc_stats = pd.read_csv('cadc_dataset_route_stats.csv', header=0, usecols=[0, 1, 2, 18, 19, 20, 21])
start_row = 36  # 25
start_frame = 80

visualize_Raw = True
visualize_DROR = True
visualize_no_dynamic = True
visualize_aggregate_cuboid_3 = True
visualize_aggregate_lidar_3 = True
visualize_aggregate_cuboid = True
visualize_aggregate_lidar = True
visualize_lidar_HPR_3DBox = False
visualize_lidar_HPR_ConvexHull = False
visualize_lidar_HPR_ProjectedKNN = False


for row in trange(start_row, len(cadc_stats)):
    # print('\n------ Date %s, Sequence %d ------' % (cadc_stats.iloc[row, 0], cadc_stats.iloc[row, 1]))
    date = cadc_stats.iloc[row, 0]
    seq = cadc_stats.iloc[row, 1]
    n_frame = cadc_stats.iloc[row, 2]

    for frame in trange(start_frame, n_frame):
        if visualize_Raw:
            lidar_raw_path = BASE + date + '/' + format(seq, '04') + \
                         "/labeled/lidar_points/data/" + format(frame, '010') + ".bin"
            lidar_raw = np.fromfile(lidar_raw_path, dtype=np.float32).reshape((-1, 4))[:, :3]
            visualize_3d(lidar_raw)

        if visualize_DROR:
            lidar_dror_path = BASE_mod + date + '/' + format(seq, '04') + \
                              "/labeled/lidar_points/lidar_dror/" + format(frame, '010') + ".npy"
            lidar_dror = np.load(lidar_dror_path).reshape(-1, 4)[:, :3]
            visualize_3d(lidar_dror)

        if visualize_no_dynamic:
            lidar_no_mobile_path = BASE_mod + date + '/' + format(seq, '04') + \
                                   "/labeled/lidar_points/lidar_no_mobile/" + format(frame, '010') + ".npy"
            lidar_no_mobile = np.load(lidar_no_mobile_path).reshape(-1, 3)
            visualize_3d(lidar_no_mobile)

        if visualize_aggregate_cuboid_3:
            cuboid_aggregated_3_path = BASE_mod + date + '/' + format(seq, '04') + \
                                     "/labeled/lidar_points/cuboid_aggregated_3/" + format(frame, '010') + ".npy"
            cuboid_aggregated_3 = np.load(cuboid_aggregated_3_path).reshape(-1, 3)
            visualize_3d(cuboid_aggregated_3)

        if visualize_aggregate_lidar_3:
            lidar_aggregated_3_path = BASE_mod + date + '/' + format(seq, '04') + \
                                      "/labeled/lidar_points/lidar_aggregated_3/" + format(frame, '010') + ".npy"
            lidar_aggregated_3 = np.load(lidar_aggregated_3_path).reshape(-1, 3)
            visualize_3d(lidar_aggregated_3)

        if visualize_aggregate_cuboid:
            cuboid_aggregated_path = BASE_mod + date + '/' + format(seq, '04') + \
                                   "/labeled/lidar_points/cuboid_aggregated/" + format(frame, '010') + ".npy"
            cuboid_aggregated = np.load(cuboid_aggregated_path).reshape(-1, 3)
            visualize_3d(cuboid_aggregated)

        if visualize_aggregate_lidar:
            lidar_aggregated_path = BASE_mod + date + '/' + format(seq, '04') + \
                                   "/labeled/lidar_points/lidar_aggregated/" + format(frame, '010') + ".npy"
            lidar_aggregated = np.load(lidar_aggregated_path).reshape(-1, 3)
            visualize_3d(lidar_aggregated)

        if visualize_lidar_HPR_3DBox:
            lidar_HPR_3DBox_path = BASE_mod + date + '/' + format(seq, '04') + \
                                   "/labeled/lidar_points/lidar_HPR_3DBox/" + format(frame, '010') + ".npy"
            lidar_HPR_3DBox = np.load(lidar_HPR_3DBox_path)
            visualize_3d(lidar_HPR_3DBox)

        if visualize_lidar_HPR_ConvexHull:
            lidar_HPR_ConvexHull_path = BASE_mod + date + '/' + format(seq, '04') + \
                                   "/labeled/lidar_points/lidar_HPR_ConvexHull/" + format(frame, '010') + ".npy"
            lidar_HPR_ConvexHull = np.load(lidar_HPR_ConvexHull_path)
            visualize_3d(lidar_HPR_ConvexHull)

        if visualize_lidar_HPR_ProjectedKNN:
            lidar_HPR_ProjectedKNN_path = BASE_mod + date + '/' + format(seq, '04') + \
                                   "/labeled/lidar_points/lidar_HPR_ProjectedKNN/" + format(frame, '010') + ".npy"
            lidar_HPR_ProjectedKNN = np.load(lidar_HPR_ProjectedKNN_path)
            visualize_3d(lidar_HPR_ProjectedKNN)
