from tqdm import *
import os
import numpy as np
import pandas as pd
from cadc_utils import load_annotation, load_novatel_data, convert_novatel_to_pose, load_calibration
from filter_lidar_dror import DROR_filter
from filter_lidar_dynamic_objects import lidar_mobile_objects_filter
from aggregate_cuboid import cuboid_aggregate, agg_cuboid_to_frame
from aggregate_lidar import lidar_aggregate
from filter_lidar_occlusion_culling_cam0 import occlusion_culling_filter

BASE = "/home/datasets/CADC/cadcd/"
BASE_mod = "/home/datasets_mod/CADC/cadcd/"
cadc_stats = pd.read_csv('cadc_dataset_route_stats.csv', header=0, usecols=[0, 1, 2, 18, 19, 20, 21])
start_row = 0
n_aggregate = 11    # 5, 11, 21, 31...
assert n_aggregate % 2 == 1

generate_DROR = False
generate_no_dynamic = False
aggregate_cuboid = False
aggregate_lidar = True


for row in trange(start_row, len(cadc_stats)):
    # print('\n------ Date %s, Sequence %d ------' % (cadc_stats.iloc[row, 0], cadc_stats.iloc[row, 1]))
    date = cadc_stats.iloc[row, 0]
    seq = cadc_stats.iloc[row, 1]
    n_frame = cadc_stats.iloc[row, 2]

    # Load calibration file
    calib_path = BASE + date + '/' + "calib"
    calib = load_calibration(calib_path)

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

    if aggregate_cuboid:
        lidars_dror = []
        for frame in range(n_frame):
            # print('Loading lidar %d of %d (%.0f%%)' % (i + 1, len(annotations), (i + 1) / len(annotations) * 100))
            lidar_dror_path = BASE_mod + date + '/' + format(seq, '04') + "/labeled/lidar_points/lidar_dror/" + \
                              format(frame, '010') + ".npy"
            lidar_dror = np.load(lidar_dror_path).reshape((-1, 4))
            lidars_dror.append(lidar_dror)

        agg_cuboids = cuboid_aggregate(lidars_dror, annotations)

        agg_cuboids_path = BASE_mod + date + '/' + format(seq, '04') + \
                               "/labeled/lidar_points/cuboid_aggregated/agg_cuboids.npy"
        if not (os.path.exists(agg_cuboids_path[:-15])):
            os.makedirs(agg_cuboids_path[:-15])
        np.save(agg_cuboids_path, agg_cuboids)

        # agg_cuboids = np.load(agg_cuboids_path, allow_pickle=True).item()
        for frame in trange(n_frame):
            cuboid_aggregated = agg_cuboid_to_frame(agg_cuboids, annotations[frame])

            cuboid_aggregated_path = BASE_mod + date + '/' + format(seq, '04') + \
                                   "/labeled/lidar_points/cuboid_aggregated/" + format(frame, '010') + ".npy"
            if not (os.path.exists(cuboid_aggregated_path[:-14])):
                os.makedirs(cuboid_aggregated_path[:-14])
            np.save(cuboid_aggregated_path, cuboid_aggregated)

    if aggregate_lidar:
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
            max_frame = min(len(lidars_no_mobile), frame + (n_aggregate - 1) // 2 + 1)

            lidar_aggregated = lidar_aggregate(lidars_no_mobile[min_frame:max_frame], cuboid_aggregated,
                                               poses[min_frame:max_frame])

            lidar_aggregated_path = BASE_mod + date + '/' + format(seq, '04') + \
                                   "/labeled/lidar_points/lidar_aggregated/" + format(frame, '010') + ".npy"
            if not (os.path.exists(lidar_aggregated_path[:-14])):
                os.makedirs(lidar_aggregated_path[:-14])
            np.save(lidar_aggregated_path, lidar_aggregated)

            lidar_agg_culling = lidar_aggregated[occlusion_culling_filter(lidar_aggregated, annotations[frame], calib)]

            lidar_agg_culling_path = BASE_mod + date + '/' + format(seq, '04') + \
                                   "/labeled/lidar_points/lidar_agg_culling/" + format(frame, '010') + ".npy"
            if not (os.path.exists(lidar_agg_culling_path[:-14])):
                os.makedirs(lidar_agg_culling_path[:-14])
            np.save(lidar_agg_culling_path, lidar_agg_culling)

    # if agg_lidar_culling:
    #     for frame in trange(n_frame):
    #         lidar_aggregated_path = BASE_mod + date + '/' + format(seq, '04') + \
    #                                 "/labeled/lidar_points/lidar_aggregated/" + format(frame, '010') + ".npy"
    #         lidar_aggregated = np.load(lidar_aggregated_path)
    #
    #         lidar_agg_culling = lidar_aggregated[occlusion_culling_filter(lidar_aggregated, annotations[frame], calib)]
    #
    #         lidar_agg_culling_path = BASE_mod + date + '/' + format(seq, '04') + \
    #                                  "/labeled/lidar_points/lidar_agg_culling/" + format(frame, '010') + ".npy"
    #         if not (os.path.exists(lidar_agg_culling_path[:-14])):
    #             os.makedirs(lidar_agg_culling_path[:-14])
    #         np.save(lidar_agg_culling_path, lidar_agg_culling)
