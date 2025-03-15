from tqdm import *
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import open3d as o3d

from cadc_utils import *
from filter_lidar_HPR_3DBox import HPR_3DBox


def HPR_ProjectedKNN(xyz=None, uv=None, img_size=None, K=51, alpha='mean'):
    # PC2VISIBILITY estimates the visibility of a 3D points cloud XYZ [m x 3]
    # associated with its projection coordinates UV [m x 2] of a point of view.
    # IMG_SIZE is the size of the image in which the point cloud is being
    # projected. K (optionnal) is the size of the neighborhood computed for
    # each point.

    # Usage:
    # - [labels] = pc2visibility(xyz, uv, img_size [, K]);
    # - [labels, depth_map] = pc2visibility(xyz, uv, img_size [, K]);

    # Copyright (c) 2018 LaBRI, Bordeaux, France.
    # Author: Pierre Biasutti
    # Email:  pierre.biasutti@labri.fr

    uv = np.floor(uv).astype(int)

    # Computing depths towards optical center
    d = np.sqrt(np.sum(xyz ** 2, 1))

    # Creating Depth-map
    dm = np.zeros(img_size)
    sd = np.sort(d)[::-1]
    si = np.argsort(d)[::-1]
    dm[uv[si,1], uv[si,0]] = sd

    # Isolating defined pixels
    unique_uv = np.unique(uv, axis=0)
    unique_d = dm[unique_uv[:,1], unique_uv[:,0]]

    # Conversion map
    unique_idx = np.arange(len(unique_uv))
    idx_map = np.zeros(dm.shape)
    idx_map[unique_uv[:,1], unique_uv[:,0]] = unique_idx
    idx_map = idx_map.astype(int)

    # KNN search
    KNN = NearestNeighbors(n_neighbors=K).fit(unique_uv).kneighbors(unique_uv, return_distance=False)
    s_KNN = np.sort(unique_d[KNN], 1)

    # Computing distance difference between local and neighbors
    distances = np.zeros(len(xyz))
    distances_lambda = distances.copy()
    for i in np.arange(len(uv)):
        distances[i] = d[i] - s_KNN[idx_map[uv[i, 1], uv[i, 0]], 0]
        distances_lambda[i] = s_KNN[idx_map[uv[i, 1], uv[i, 0]], -1] - s_KNN[idx_map[uv[i, 1], uv[i, 0]], 0]

    visibility = np.exp(- (distances / distances_lambda) ** 2)

    if alpha == 'mean':
        filtered = np.flatnonzero(visibility >= np.mean(visibility))
    elif alpha == 'median':
        filtered = np.flatnonzero(visibility >= np.median(visibility))
    else:
        filtered = np.flatnonzero(visibility >= alpha)
    return filtered


if __name__ == '__main__':
    BASE = "/home/datasets/CADC/cadcd/"
    BASE_mod = "/home/datasets_mod/CADC/cadcd/"
    cadc_stats = pd.read_csv('cadc_dataset_route_stats.csv', header=0, usecols=[0, 1, 2, 18, 19, 20, 21])
    start_row = 36  # 32
    start_frame = 80    # 30

    for row in trange(start_row, len(cadc_stats)):
        date = cadc_stats.iloc[row, 0]
        seq = cadc_stats.iloc[row, 1]
        n_frame = cadc_stats.iloc[row, 2]

        calib_path = BASE + date + '/' + "calib"
        calib = load_calibration(calib_path)
        T_IMG_CAM = np.eye(4)
        T_IMG_CAM[0:3, 0:3] = np.array(calib['CAM00']['camera_matrix']['data']).reshape(-1, 3)
        T_IMG_CAM = T_IMG_CAM[0:3, 0:4]  # remove last row
        T_CAM_LIDAR = np.linalg.inv(np.array(calib['extrinsics']['T_LIDAR_CAM00']))

        annotations_path = BASE + date + '/' + format(seq, '04') + "/3d_ann.json"

        for frame in trange(start_frame, n_frame):
            path = BASE_mod + date + '/' + format(seq, '04') + \
                   "/labeled/lidar_points/lidar_aggregated/" + format(frame, '010') + ".npy"
            points = np.load(path).reshape(-1, 3)
            lidar2cam0 = Lidar2Cam(points, T_IMG_CAM, T_CAM_LIDAR)
            x = lidar2cam0[:, 0]
            y = lidar2cam0[:, 1]
            depth = lidar2cam0[:, 2]
            points_cam0 = points[np.logical_and.reduce((x >= 0, x < w, y >= 0, y < h, depth >= 0))]
            # visualize_3d(points_cam0)

            # annotation = load_annotation(annotations_path)[frame]
            # filtered_culling = HPR_3DBox(points_cam0, annotation, calib)
            # visualize_3d(points_cam0[filtered_culling])

            project_cam0 = CropPoints(lidar2cam0)
            filtered_HPR_ProjectedKNN = HPR_ProjectedKNN(points_cam0, project_cam0[:, :2], (h, w), K=51, alpha='mean') # to test 0.99
            deleted_HPR_ProjectedKNN = list(set(range(len(points_cam0))) - set(filtered_HPR_ProjectedKNN))
            # visualize_3d(points_cam0[filtered_HPR_ProjectedKNN])

            # lidar_3d = o3d.geometry.PointCloud()
            # lidar_3d.points = o3d.utility.Vector3dVector(points_cam0)
            # lidar_3d.colors = o3d.utility.Vector3dVector(np.ones((len(points_cam0), 3)) * 0.5)
            # np.asarray(lidar_3d.colors)[deleted_HPR_ProjectedKNN] = [1, 0, 0]
            # o3d.visualization.draw_geometries([lidar_3d])

            # Generate depth map (H, W) from points (N, 3)
            depth_aggregate = GenerateDepth(project_cam0[filtered_HPR_ProjectedKNN])
            plt.imshow(depth_aggregate.clip(0,80), cmap='jet', vmin=0, vmax=80)
            plt.show()
