from tqdm import *
import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull
import open3d as o3d

from cadc_utils import *
from filter_lidar_HPR_3DBox import HPR_3DBox


def HPR_ConvexHull(points, param=4):
    n = len(points)  # total n points
    points = points - np.repeat(np.array([[0, 0, 0]]), n, axis=0)  # Move C to the origin
    normPoints = np.linalg.norm(points, axis=1)  # Normed points, sqrt(x^2 + y^2 + (z-100)^2)
    R = np.repeat(max(normPoints) * np.power(10.0, param), n, axis=0)  # Radius of Sphere

    flippedPointsTemp = 2 * np.multiply(np.repeat((R - normPoints).reshape(n, 1), len(points[0]), axis=1), points)
    flippedPoints = np.divide(flippedPointsTemp, np.repeat(normPoints.reshape(n, 1), len(points[0]),
                                                           axis=1))  # Apply Equation to get Flipped Points
    flippedPoints += points

    plusOrigin = np.append(flippedPoints, [[0, 0, 0]], axis=0)  # All points plus origin
    hull = ConvexHull(plusOrigin)  # Visibal points plus possible origin. Use its vertices property.

    filtered = hull.vertices[:-1]
    return filtered


if __name__ == '__main__':
    BASE = "/home/datasets/CADC/cadcd/"
    BASE_mod = "/home/datasets_mod/CADC/cadcd/"
    cadc_stats = pd.read_csv('cadc_dataset_route_stats.csv', header=0, usecols=[0, 1, 2, 18, 19, 20, 21])
    start_row = 36  # 32
    start_frame = 80  # 30

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
            visualize_3d(points_cam0)

            # annotation = load_annotation(annotations_path)[frame]
            # filtered_culling = HPR_3DBox(points_cam0, annotation, calib)
            # visualize_3d(points_cam0[filtered_culling])

            filtered_HPR_ConvexHull = HPR_ConvexHull(points_cam0, param=4)
            deleted_HPR_ConvexHull = list(set(range(len(points_cam0))) - set(filtered_HPR_ConvexHull))
            visualize_3d(points_cam0[filtered_HPR_ConvexHull])

            # lidar_3d = o3d.geometry.PointCloud()
            # lidar_3d.points = o3d.utility.Vector3dVector(points_cam0)
            # lidar_3d.colors = o3d.utility.Vector3dVector(np.ones((len(points_cam0), 3)) * 0.5)
            # np.asarray(lidar_3d.colors)[deleted_HPR_ConvexHull] = [1, 0, 0]
            # o3d.visualization.draw_geometries([lidar_3d])

            # Generate depth map (H, W) from points (N, 3)
            # project_cam0 = CropPoints(lidar2cam0)
            # depth_aggregate = GenerateDepth(project_cam0[filtered_HPR_ConvexHull])
            # plt.imshow(depth_aggregate.clip(0,80), cmap='jet', vmin=0, vmax=80)
            # plt.show()
