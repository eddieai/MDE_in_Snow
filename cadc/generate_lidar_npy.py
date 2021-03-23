from tqdm import *
import os
import numpy as np
import pandas as pd
from cadc_utils import *
from filter_lidar_dror import DROR_filter
from filter_lidar_dynamic_objects import lidar_mobile_objects_filter
from aggregate_cuboid import cuboid_aggregate
from aggregate_lidar import lidar_aggregate
from filter_lidar_HPR_3DBox import HPR_3DBox
from filter_lidar_HPR_ConvexHull import HPR_ConvexHull
from filter_lidar_HPR_ProjectedKNN import HPR_ProjectedKNN

BASE = "/home/datasets/CADC/cadcd/"
BASE_mod = "/home/datasets_mod/CADC/cadcd/"
cadc_stats = pd.read_csv('cadc_dataset_route_stats.csv', header=0, usecols=[0, 1, 2, 18, 19, 20, 21])
start_row = 0

generate_DROR = False
generate_no_dynamic = False
aggregate_cuboid_3 = True
aggregate_lidar_3 = True
aggregate_cuboid = True
aggregate_lidar = True
aggregate_HPR_ProjectedKNN = True
aggregate_HPR_ProjectedKNN_99 = True
aggregate_HPR_ConvexHull = True
aggregate_HPR_3DBox = False  # not yet relaunched after correcting dynamic object aggregation


for row in trange(start_row, len(cadc_stats)):
    # print('\n------ Date %s, Sequence %d ------' % (cadc_stats.iloc[row, 0], cadc_stats.iloc[row, 1]))
    date = cadc_stats.iloc[row, 0]
    seq = cadc_stats.iloc[row, 1]
    n_frame = cadc_stats.iloc[row, 2]

    # Load calibration file
    calib_path = BASE + date + '/' + "calib"
    calib = load_calibration(calib_path)
    # Projection matrix from lidar to camera (extrinsics)
    T_CAM_LIDAR = np.linalg.inv(np.array(calib['extrinsics']['T_LIDAR_CAM00']))[:3, :4]
    # Projection matrix from camera to image (intrinsics)
    T_IMG_CAM = np.array(calib['CAM00']['camera_matrix']['data']).reshape(3, 3)

    # Load 3d annotations
    annotations_path = BASE + date + '/' + format(seq, '04') + "/3d_ann.json"
    annotations = load_annotation(annotations_path)

    # Load novatel IMU data
    novatel_path = BASE + date + '/' + format(seq, '04') + "/labeled/novatel/data/"
    novatel = load_novatel_data(novatel_path)
    poses = convert_novatel_to_pose(novatel)

    if generate_DROR:
        for frame in trange(n_frame):
            # print('Loading lidar %d of %d (%.0f%%)' % (i + 1, n_frame, (i + 1) / n_frame * 100))
            lidar_raw_path = BASE + date + '/' + format(seq, '04') + \
                         "/labeled/lidar_points/data/" + format(frame, '010') + ".bin"
            lidar = np.fromfile(lidar_raw_path, dtype=np.float32).reshape((-1, 4))

            lidar_dror = lidar[DROR_filter(lidar)]

            lidar_dror_path = BASE_mod + date + '/' + format(seq, '04') + \
                              "/labeled/lidar_points/lidar_dror/" + format(frame, '010') + ".npy"
            if not (os.path.exists(lidar_dror_path[:-14])):
                os.makedirs(lidar_dror_path[:-14])
            np.save(lidar_dror_path, lidar_dror)

    if generate_no_dynamic:
        for frame in trange(n_frame):
            lidar_dror_path = BASE_mod + date + '/' + format(seq, '04') + \
                         "/labeled/lidar_points/lidar_dror/" + format(frame, '010') + ".npy"
            lidar_dror = np.load(lidar_dror_path).reshape((-1, 4))[:, :3]

            lidar_no_mobile = lidar_dror[lidar_mobile_objects_filter(lidar_dror, annotations[frame])]

            lidar_no_mobile_path = BASE_mod + date + '/' + format(seq, '04') + \
                                   "/labeled/lidar_points/lidar_no_mobile/" + format(frame, '010') + ".npy"
            if not (os.path.exists(lidar_no_mobile_path[:-14])):
                os.makedirs(lidar_no_mobile_path[:-14])
            np.save(lidar_no_mobile_path, lidar_no_mobile)

    if aggregate_cuboid_3:
        n_aggregate = 3  # 5, 11, 21, 31...
        assert n_aggregate % 2 == 1
        lidars_dror = []
        for frame in range(n_frame):
            # print('Loading lidar %d of %d (%.0f%%)' % (i + 1, len(annotations), (i + 1) / len(annotations) * 100))
            lidar_dror_path = BASE_mod + date + '/' + format(seq, '04') + "/labeled/lidar_points/lidar_dror/" + \
                              format(frame, '010') + ".npy"
            lidar_dror = np.load(lidar_dror_path).reshape((-1, 4))
            lidars_dror.append(lidar_dror)

        for frame in trange(n_frame):
            min_frame = max(0, frame - (n_aggregate - 1) // 2)
            max_frame = min(n_frame, frame + (n_aggregate - 1) // 2 + 1)
            ref_frame = min(frame, (n_aggregate - 1) // 2)

            cuboid_aggregated = cuboid_aggregate(ref_frame, lidars_dror[min_frame:max_frame], annotations[min_frame:max_frame])

            cuboid_aggregated_path = BASE_mod + date + '/' + format(seq, '04') + \
                                   "/labeled/lidar_points/cuboid_aggregated_3/" + format(frame, '010') + ".npy"
            if not (os.path.exists(cuboid_aggregated_path[:-14])):
                os.makedirs(cuboid_aggregated_path[:-14])
            np.save(cuboid_aggregated_path, cuboid_aggregated)

    if aggregate_lidar_3:
        n_aggregate = 3  # 5, 11, 21, 31...
        assert n_aggregate % 2 == 1
        lidars_no_mobile = []
        for frame in range(n_frame):
            lidar_no_mobile_path = BASE_mod + date + '/' + format(seq, '04') \
                                   + "/labeled/lidar_points/lidar_no_mobile/" + \
                                   format(frame, '010') + ".npy"
            lidar_no_mobile = np.load(lidar_no_mobile_path).reshape((-1, 3))
            lidars_no_mobile.append(lidar_no_mobile)

        for frame in trange(n_frame):
            # print('Aggregating frame %d of %d (%.0f%%)' % (target_frame+1, len(annotation), (target_frame+1)/len(annotation)*100))
            cuboid_aggregated_path = BASE_mod + date + '/' + format(seq, '04') \
                                    + "/labeled/lidar_points/cuboid_aggregated_3/" + \
                                    format(frame, '010') + ".npy"
            cuboid_aggregated = np.load(cuboid_aggregated_path).reshape((-1, 3))

            min_frame = max(0, frame - (n_aggregate - 1) // 2)
            max_frame = min(n_frame, frame + (n_aggregate - 1) // 2 + 1)
            ref_frame = min(frame, (n_aggregate - 1) // 2)

            lidar_aggregated = lidar_aggregate(ref_frame, lidars_no_mobile[min_frame:max_frame], cuboid_aggregated,
                                               poses[min_frame:max_frame])

            lidar_aggregated_path = BASE_mod + date + '/' + format(seq, '04') + \
                                   "/labeled/lidar_points/lidar_aggregated_3/" + format(frame, '010') + ".npy"
            if not (os.path.exists(lidar_aggregated_path[:-14])):
                os.makedirs(lidar_aggregated_path[:-14])
            np.save(lidar_aggregated_path, lidar_aggregated)

    if aggregate_cuboid:
        n_aggregate = 11  # 5, 11, 21, 31...
        assert n_aggregate % 2 == 1
        lidars_dror = []
        for frame in range(n_frame):
            # print('Loading lidar %d of %d (%.0f%%)' % (i + 1, len(annotations), (i + 1) / len(annotations) * 100))
            lidar_dror_path = BASE_mod + date + '/' + format(seq, '04') + "/labeled/lidar_points/lidar_dror/" + \
                              format(frame, '010') + ".npy"
            lidar_dror = np.load(lidar_dror_path).reshape((-1, 4))
            lidars_dror.append(lidar_dror)

        for frame in trange(n_frame):
            min_frame = max(0, frame - (n_aggregate - 1) // 2)
            max_frame = min(n_frame, frame + (n_aggregate - 1) // 2 + 1)
            ref_frame = min(frame, (n_aggregate - 1) // 2)

            cuboid_aggregated = cuboid_aggregate(ref_frame, lidars_dror[min_frame:max_frame], annotations[min_frame:max_frame])

            cuboid_aggregated_path = BASE_mod + date + '/' + format(seq, '04') + \
                                   "/labeled/lidar_points/cuboid_aggregated/" + format(frame, '010') + ".npy"
            if not (os.path.exists(cuboid_aggregated_path[:-14])):
                os.makedirs(cuboid_aggregated_path[:-14])
            np.save(cuboid_aggregated_path, cuboid_aggregated)

    if aggregate_lidar:
        n_aggregate = 11  # 5, 11, 21, 31...
        assert n_aggregate % 2 == 1
        lidars_no_mobile = []
        for frame in range(n_frame):
            lidar_no_mobile_path = BASE_mod + date + '/' + format(seq, '04') \
                                   + "/labeled/lidar_points/lidar_no_mobile/" + \
                                   format(frame, '010') + ".npy"
            lidar_no_mobile = np.load(lidar_no_mobile_path).reshape((-1, 3))
            lidars_no_mobile.append(lidar_no_mobile)

        for frame in trange(n_frame):
            # print('Aggregating frame %d of %d (%.0f%%)' % (target_frame+1, len(annotation), (target_frame+1)/len(annotation)*100))
            cuboid_aggregated_path = BASE_mod + date + '/' + format(seq, '04') \
                                    + "/labeled/lidar_points/cuboid_aggregated/" + \
                                    format(frame, '010') + ".npy"
            cuboid_aggregated = np.load(cuboid_aggregated_path).reshape((-1, 3))

            min_frame = max(0, frame - (n_aggregate - 1) // 2)
            max_frame = min(n_frame, frame + (n_aggregate - 1) // 2 + 1)
            ref_frame = min(frame, (n_aggregate - 1) // 2)

            lidar_aggregated = lidar_aggregate(ref_frame, lidars_no_mobile[min_frame:max_frame], cuboid_aggregated,
                                               poses[min_frame:max_frame])

            lidar_aggregated_path = BASE_mod + date + '/' + format(seq, '04') + \
                                   "/labeled/lidar_points/lidar_aggregated/" + format(frame, '010') + ".npy"
            if not (os.path.exists(lidar_aggregated_path[:-14])):
                os.makedirs(lidar_aggregated_path[:-14])
            np.save(lidar_aggregated_path, lidar_aggregated)

    if aggregate_HPR_ProjectedKNN:
        for frame in trange(n_frame):
            lidar_aggregated_path = BASE_mod + date + '/' + format(seq, '04') + \
                                    "/labeled/lidar_points/lidar_aggregated/" + format(frame, '010') + ".npy"
            lidar_aggregated = np.load(lidar_aggregated_path)

            lidar2cam0 = Lidar2Cam(lidar_aggregated, T_IMG_CAM, T_CAM_LIDAR)
            x = lidar2cam0[:, 0]
            y = lidar2cam0[:, 1]
            depth = lidar2cam0[:, 2]
            points_cam0 = lidar_aggregated[np.logical_and.reduce((x >= 0, x < w, y >= 0, y < h, depth >= 0))]
            project_cam0 = CropPoints(lidar2cam0)
            lidar_HPR_ProjectedKNN = points_cam0[HPR_ProjectedKNN(points_cam0, project_cam0[:, :2], (h, w), K=51, alpha='mean')]

            lidar_HPR_ProjectedKNN_path = BASE_mod + date + '/' + format(seq, '04') + \
                                     "/labeled/lidar_points/lidar_HPR_ProjectedKNN/" + format(frame, '010') + ".npy"
            if not (os.path.exists(lidar_HPR_ProjectedKNN_path[:-14])):
                os.makedirs(lidar_HPR_ProjectedKNN_path[:-14])
            np.save(lidar_HPR_ProjectedKNN_path, lidar_HPR_ProjectedKNN)

    if aggregate_HPR_ProjectedKNN_99:
        for frame in trange(n_frame):
            lidar_aggregated_path = BASE_mod + date + '/' + format(seq, '04') + \
                                    "/labeled/lidar_points/lidar_aggregated/" + format(frame, '010') + ".npy"
            lidar_aggregated = np.load(lidar_aggregated_path)

            lidar2cam0 = Lidar2Cam(lidar_aggregated, T_IMG_CAM, T_CAM_LIDAR)
            x = lidar2cam0[:, 0]
            y = lidar2cam0[:, 1]
            depth = lidar2cam0[:, 2]
            points_cam0 = lidar_aggregated[np.logical_and.reduce((x >= 0, x < w, y >= 0, y < h, depth >= 0))]
            project_cam0 = CropPoints(lidar2cam0)
            lidar_HPR_ProjectedKNN = points_cam0[
                HPR_ProjectedKNN(points_cam0, project_cam0[:, :2], (h, w), K=51, alpha=0.99)]

            lidar_HPR_ProjectedKNN_path = BASE_mod + date + '/' + format(seq, '04') + \
                                          "/labeled/lidar_points/lidar_HPR_ProjectedKNN_0.99/" + format(frame,
                                                                                                   '010') + ".npy"
            if not (os.path.exists(lidar_HPR_ProjectedKNN_path[:-14])):
                os.makedirs(lidar_HPR_ProjectedKNN_path[:-14])
            np.save(lidar_HPR_ProjectedKNN_path, lidar_HPR_ProjectedKNN)

    if aggregate_HPR_ConvexHull:
        for frame in trange(n_frame):
            lidar_aggregated_path = BASE_mod + date + '/' + format(seq, '04') + \
                                    "/labeled/lidar_points/lidar_aggregated/" + format(frame, '010') + ".npy"
            lidar_aggregated = np.load(lidar_aggregated_path)

            lidar2cam0 = Lidar2Cam(lidar_aggregated, T_IMG_CAM, T_CAM_LIDAR)
            x = lidar2cam0[:, 0]
            y = lidar2cam0[:, 1]
            depth = lidar2cam0[:, 2]
            points_cam0 = lidar_aggregated[np.logical_and.reduce((x >= 0, x < w, y >= 0, y < h, depth >= 0))]
            lidar_HPR_ConvexHull = points_cam0[HPR_ConvexHull(points_cam0, param=4)]

            lidar_HPR_ConvexHull_path = BASE_mod + date + '/' + format(seq, '04') + \
                                        "/labeled/lidar_points/lidar_HPR_ConvexHull/" + format(frame,
                                                                                               '010') + ".npy"
            if not (os.path.exists(lidar_HPR_ConvexHull_path[:-14])):
                os.makedirs(lidar_HPR_ConvexHull_path[:-14])
            np.save(lidar_HPR_ConvexHull_path, lidar_HPR_ConvexHull)

    if aggregate_HPR_3DBox:
        for frame in trange(n_frame):
            lidar_aggregated_path = BASE_mod + date + '/' + format(seq, '04') + \
                                    "/labeled/lidar_points/lidar_aggregated/" + format(frame, '010') + ".npy"
            lidar_aggregated = np.load(lidar_aggregated_path)

            lidar_HPR_3DBox = lidar_aggregated[HPR_3DBox(lidar_aggregated, annotations[frame], calib)]

            lidar_HPR_3DBox_path = BASE_mod + date + '/' + format(seq, '04') + \
                                     "/labeled/lidar_points/lidar_HPR_3DBox/" + format(frame, '010') + ".npy"
            if not (os.path.exists(lidar_HPR_3DBox_path[:-14])):
                os.makedirs(lidar_HPR_3DBox_path[:-14])
            np.save(lidar_HPR_3DBox_path, lidar_HPR_3DBox)
