#!/usr/bin/env python
import json
import numpy as np
from scipy.spatial import ConvexHull
from matplotlib.path import Path
import matplotlib.pyplot as plt
from cadc_utils import load_calibration
from scipy.spatial.transform import Rotation as R


def depthmap_mobile_objects_filter(annotation, T_IMG_CAM, T_CAM_LIDAR):
    img_h, img_w = 1024, 1280
    x, y = np.meshgrid(np.arange(img_w), np.arange(img_h))  # make a canvas with coordinates
    x, y = x.flatten(), y.flatten()
    pixels = np.vstack((x, y)).T
    mask = np.zeros((img_h, img_w), dtype=int)

    # Add each cuboid to image
    for cuboid in annotation['cuboids']:
        T_Lidar_Cuboid = np.eye(4)
        T_Lidar_Cuboid[0:3, 0:3] = R.from_euler('z', cuboid['yaw'], degrees=False).as_dcm()
        T_Lidar_Cuboid[0][3] = cuboid['position']['x']
        T_Lidar_Cuboid[1][3] = cuboid['position']['y']
        T_Lidar_Cuboid[2][3] = cuboid['position']['z']

        width = cuboid['dimensions']['x']
        length = cuboid['dimensions']['y']
        height = cuboid['dimensions']['z']

        cuboid_vertices = np.array([
            [length / 2, -width / 2, -height / 2, 1],
            [length / 2, -width / 2, height / 2, 1],
            [length / 2, width / 2, -height / 2, 1],
            [length / 2, width / 2, height / 2, 1],
            [-length / 2, -width / 2, -height / 2, 1],
            [-length / 2, -width / 2, height / 2, 1],
            [-length / 2, width / 2, -height / 2, 1],
            [-length / 2, width / 2, height / 2, 1]
        ])
        t_lidar_cuboid = np.dot(cuboid_vertices, T_Lidar_Cuboid.T)
        t_cam_cuboid = np.dot(t_lidar_cuboid, T_CAM_LIDAR.T)
        t_img_cuboid = np.dot(t_cam_cuboid, T_IMG_CAM.T)

        cuboid_projected_vertices = np.stack([t_img_cuboid[:, 0] / t_img_cuboid[:, 2],
                                              t_img_cuboid[:, 1] / t_img_cuboid[:, 2],
                                              t_img_cuboid[:, 2]], axis=-1)

        if np.any(cuboid_projected_vertices[:, 2] < 0):
            continue

        hull = ConvexHull(cuboid_projected_vertices[:, :2])
        polygon = Path(cuboid_projected_vertices[hull.vertices][:, :2])
        inside_polygon = polygon.contains_points(pixels)
        mask = np.logical_or(mask, inside_polygon.reshape(img_h, img_w))

    return mask

if __name__ == '__main__':

    BASE = "/home/datasets/CADC/cadcd/"
    date = '2019_02_27'
    seq = 10
    start_frame = 33
    cam = 0

    annotations_file = BASE + date + '/' + format(seq, '04') + "/3d_ann.json"
    # Load 3d annotations
    with open(annotations_file) as f:
        annotations_data = json.load(f)

    calib_path = BASE + date + '/' + "calib"
    calib = load_calibration(calib_path)
    # Projection matrix from camera to image frame
    T_IMG_CAM = np.eye(4)
    T_IMG_CAM[0:3, 0:3] = np.array(calib['CAM0' + str(cam)]['camera_matrix']['data']).reshape(-1, 3)
    T_IMG_CAM = T_IMG_CAM[0:3, 0:4]  # remove last row
    T_CAM_LIDAR = np.linalg.inv(np.array(calib['extrinsics']['T_LIDAR_CAM0' + str(cam)]))

    plt.ion()
    plt.figure()

    for frame in range(start_frame, len(annotations_data)):
        mask = depthmap_mobile_objects_filter(annotations_data[frame], T_IMG_CAM, T_CAM_LIDAR)

        img_path = BASE + date + '/' + format(seq, '04') + "/labeled" + "/image_0" + str(cam) + "/data/" + format(frame, '010') + ".png"
        img = plt.imread(img_path)
        img[mask] = 1
        plt.imshow(img)
        plt.pause(0.1)
        plt.show()
