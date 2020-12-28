from tqdm import *
import numpy as np
import pandas as pd
import open3d as o3d


BASE = "/home/datasets/CADC/cadcd/"
BASE_mod = "/home/datasets_mod/CADC/cadcd/"
cadc_stats = pd.read_csv('cadc_dataset_route_stats.csv', header=0, usecols=[0, 1, 2, 18, 19, 20, 21])
start_row = 0  # 25
start_frame = 5

visualize_Raw = False
visualize_DROR = True
visualize_no_dynamic = False
visualize_aggregate_cuboid = False
visualize_aggregate_lidar = False
visualize_agg_lidar_culling = False


def visualize_3d(lidar):
    lidar_3d = o3d.geometry.PointCloud()
    lidar_3d.points = o3d.utility.Vector3dVector(lidar)
    lidar_3d.colors = o3d.utility.Vector3dVector(np.ones((len(lidar), 3)) * 0.5)
    o3d.visualization.draw_geometries([lidar_3d])


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

        if visualize_agg_lidar_culling:
            lidar_agg_culling_path = BASE_mod + date + '/' + format(seq, '04') + \
                                   "/labeled/lidar_points/lidar_agg_culling/" + format(frame, '010') + ".npy"
            lidar_agg_culling = np.load(lidar_agg_culling_path)
            visualize_3d(lidar_agg_culling)
