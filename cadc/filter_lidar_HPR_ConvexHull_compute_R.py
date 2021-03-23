from tqdm import *
import numpy as np
from scipy.spatial import ConvexHull
import open3d as o3d
import matplotlib.pyplot as plt
from cadc_utils import load_annotation, load_calibration, Lidar2Cam
from filter_lidar_HPR_3DBox import HPR_3DBox


def visualize_3d(lidar):
    lidar_3d = o3d.geometry.PointCloud()
    lidar_3d.points = o3d.utility.Vector3dVector(lidar)
    lidar_3d.colors = o3d.utility.Vector3dVector(np.ones((len(lidar), 3)) * 0.5)
    o3d.visualization.draw_geometries([lidar_3d])


def sphericalFlip(points, param, center=np.array([[0, 0, 0]])):
    n = len(points)  # total n points
    points = points - np.repeat(center, n, axis=0)  # Move C to the origin
    normPoints = np.linalg.norm(points, axis=1)  # Normed points, sqrt(x^2 + y^2 + (z-100)^2)
    R = np.repeat(max(normPoints) * np.power(10.0, param), n, axis=0)  # Radius of Sphere

    flippedPointsTemp = 2 * np.multiply(np.repeat((R - normPoints).reshape(n, 1), len(points[0]), axis=1), points)
    flippedPoints = np.divide(flippedPointsTemp, np.repeat(normPoints.reshape(n, 1), len(points[0]),
                                                           axis=1))  # Apply Equation to get Flipped Points
    flippedPoints += points

    return flippedPoints


def convexHull(flippedPoints):
    plusOrigin = np.append(flippedPoints, [[0, 0, 0]], axis=0)  # All points plus origin
    hull = ConvexHull(plusOrigin)  # Visibal points plus possible origin. Use its vertices property.

    return hull


BASE = "/home/datasets/CADC/cadcd/"
BASE_mod = "/home/datasets_mod/CADC/cadcd/"
date = '2019_02_27'
seq = 19        # to run
frame = 5
show3D = False

path = BASE_mod + date + '/' + format(seq, '04') + \
       "/labeled/lidar_points/lidar_aggregated/" + format(frame, '010') + ".npy"
points = np.load(path).reshape(-1, 3)

if show3D:
    visualize_3d(points)

annotations_path = BASE + date + '/' + format(seq, '04') + "/3d_ann.json"
annotation = load_annotation(annotations_path)[frame]

calib_path = BASE + date + '/' + "calib"
calib = load_calibration(calib_path)
T_IMG_CAM = np.eye(4)
T_IMG_CAM[0:3, 0:3] = np.array(calib['CAM00']['camera_matrix']['data']).reshape(-1, 3)
T_IMG_CAM = T_IMG_CAM[0:3, 0:4]  # remove last row
T_CAM_LIDAR = np.linalg.inv(np.array(calib['extrinsics']['T_LIDAR_CAM00']))

path = BASE_mod + date + '/' + format(seq, '04') + \
       "/labeled/lidar_points/lidar_aggregated_21/" + format(frame, '010') + ".npy"
points = np.load(path).reshape(-1, 3)
project_cam0 = Lidar2Cam(points, T_IMG_CAM, T_CAM_LIDAR)
x = project_cam0[:, 0]
y = project_cam0[:, 1]
depth = project_cam0[:, 2]
points_cam0_idx = np.nonzero(np.logical_and.reduce((x >= 0, x < 1280, y >= 0, y < 1024, depth >= 0)))[0]
points_cam0 = points[points_cam0_idx]

if show3D:
    visualize_3d(points_cam0)

annotation = load_annotation(annotations_path)[frame]
filtered_culling = HPR_3DBox(points_cam0, annotation, calib)
deleted_culling = list(set(range(len(points_cam0))) - set(filtered_culling))

if show3D:
    visualize_3d(points_cam0[filtered_culling])

precision_list = []
recall_list = []
search_range = np.arange(0, 10, 0.1)
for param in search_range:
    flippedPoints = sphericalFlip(points_cam0, param=param)  # Reflect the point cloud about a sphere centered at C
    hull = convexHull(flippedPoints)  # Take the convex hull of the sphere center and the reflected point cloud
    filtered_HPR = hull.vertices[:-1]
    deleted_HPR = list(set(range(len(points_cam0))) - set(filtered_HPR))

    if show3D:
        visualize_3d(points_cam0[filtered_HPR])

    TP = len(set(filtered_HPR) & set(filtered_culling))
    TN = len(set(deleted_HPR) & set(deleted_culling))
    FP = len(set(filtered_HPR) & set(deleted_culling))
    FN = len(set(deleted_HPR) & set(filtered_culling))
    precision = TP / (TP + FP)
    precision_list.append(precision)
    recall = TP / (TP + FN)
    recall_list.append(recall)

    print('param %.2f: points_cam0 %d \t filtered_culling %d \t filtered_HPR %d \t deleted_culling %d \t deleted_HPR %d \n'
          'TruePositive %d \t TrueNegative %d \t FalsePositive %d \t FalseNegative %d \n'
          'Precision %.2f \t Recall %.2f\n' %
          (param, len(points_cam0), len(filtered_culling), len(filtered_HPR), len(deleted_culling), len(deleted_HPR),
           TP, TN, FP, FN, precision, recall))

np.save('recall_list.npy', recall_list)
np.save('precision_list.npy', precision_list)

fig = plt.figure()
plt.plot(recall_list, precision_list, 'bo-')
for i, v in enumerate(search_range):
    plt.text(recall_list[i], precision_list[i]+0.003, "%.2f" % v, ha="center")
plt.show()
