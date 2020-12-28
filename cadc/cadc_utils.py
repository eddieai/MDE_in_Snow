import math
import os
import numpy as np
import utm
import yaml
import json

h, w = 1024, 1280


# Project lidar points to camera coordinate
def Lidar2Cam(lidar, T_IMG_CAM, T_CAM_LIDAR):
    t_cam_lidar = np.dot(np.hstack((lidar[:, :3], np.ones((len(lidar), 1)))), T_CAM_LIDAR.T)
    t_img_cam = np.dot(t_cam_lidar, T_IMG_CAM.T)
    return np.stack((t_img_cam[:, 0] / t_img_cam[:, 2],
                     t_img_cam[:, 1] / t_img_cam[:, 2],
                     t_img_cam[:, 2]), axis=-1)


# After projecting lidar points to camera coordinate, keep only points within camera RGB image range
def CropPoints(points):
    x = points[:, 0]
    y = points[:, 1]
    depth = points[:, 2]
    return points[np.logical_and.reduce((x >= 0, x < w, y >= 0, y < h, depth >= 0))]


# Generate ground truth depth map from projected lidar points
def GenerateDepth(points):
    # Aggregate by depth from far to near
    points_sorted = points[points[:, 2].argsort()][::-1]

    depth = np.zeros((h + 1, w + 1))
    depth_x = np.round(points_sorted[:, 1]).astype(int)
    depth_y = np.round(points_sorted[:, 0]).astype(int)
    depth[depth_x, depth_y] = points_sorted[:, 2]
    return depth[:h, :w]


# Converts GPS data to poses in the ENU frame
def convert_novatel_to_pose(novatel):
    poses = []

    for gps_msg in novatel:
        utm_data = utm.from_latlon(float(gps_msg[0]), float(gps_msg[1]))
        ellipsoidal_height = float(gps_msg[2]) + float(gps_msg[3]);
        roll = np.deg2rad(float(gps_msg[7]));
        pitch = np.deg2rad(float(gps_msg[8]));
        yaw = np.deg2rad(float(gps_msg[9]));

        c_phi = np.cos(roll)
        s_phi = np.sin(roll)
        c_theta = np.cos(pitch)
        s_theta = np.sin(pitch)
        c_psi = np.cos(yaw)
        s_psi = np.sin(yaw)
        poses.append([
            np.array([
                [c_psi * c_phi - s_psi * s_theta * s_phi, -s_psi * c_theta, c_psi * s_phi + s_psi * s_theta * c_phi, 0],
                [s_psi * c_phi + c_psi * s_theta * s_phi, c_psi * c_theta, s_psi * s_phi - c_psi * s_theta * c_phi, 0],
                [-c_theta * s_phi, s_theta, c_theta * c_phi, 0],
                [0, 0, 0, 1]
            ]),
            np.array([
                [1, 0, 0, -utm_data[1]],
                [0, 1, 0, utm_data[0]],
                [0, 0, 1, ellipsoidal_height],
                [0, 0, 0, 1]
            ])
        ])

    return poses


def load_novatel_data(novatel_path):
    files = os.listdir(novatel_path)
    novatel = []

    for file in sorted(files):
        with open(novatel_path + file) as fp:
            novatel.append(fp.readline().split(' '))

    return novatel


def load_annotation(annotations_path):
    with open(annotations_path) as f:
        annotation = json.load(f)

    return annotation


def load_calibration(calib_path):
    calib = {}

    # Get calibrations
    calib['extrinsics'] = yaml.load(open(calib_path + '/extrinsics.yaml'), yaml.SafeLoader)
    calib['CAM00'] = yaml.load(open(calib_path + '/00.yaml'), yaml.SafeLoader)
    calib['CAM01'] = yaml.load(open(calib_path + '/01.yaml'), yaml.SafeLoader)
    calib['CAM02'] = yaml.load(open(calib_path + '/02.yaml'), yaml.SafeLoader)
    calib['CAM03'] = yaml.load(open(calib_path + '/03.yaml'), yaml.SafeLoader)
    calib['CAM04'] = yaml.load(open(calib_path + '/04.yaml'), yaml.SafeLoader)
    calib['CAM05'] = yaml.load(open(calib_path + '/05.yaml'), yaml.SafeLoader)
    calib['CAM06'] = yaml.load(open(calib_path + '/06.yaml'), yaml.SafeLoader)
    calib['CAM07'] = yaml.load(open(calib_path + '/07.yaml'), yaml.SafeLoader)

    return calib
