from tqdm import *
import os
import numpy as np
import pandas as pd
from PIL import Image
from cadc_utils import load_annotation, load_calibration
from cadc_utils import Lidar2Cam, CropPoints, GenerateDepth

BASE = "/home/datasets/CADC/cadcd/"
BASE_mod = "/home/datasets_mod/CADC/cadcd/"
cadc_stats = pd.read_csv('cadc_dataset_route_stats.csv', header=0, usecols=[0, 1, 2, 18, 19, 20, 21])
start_row = 0

generate_Raw = False
generate_DROR = False
generate_aggregated_3 = True
generate_aggregated = True
generate_HPR_ProjectedKNN = True
generate_HPR_ProjectedKNN_99 = True
generate_HPR_ConvexHull = True
generate_HPR_3DBox = False  # not yet relaunched after correcting dynamic object aggregation
VISUALIZE = False
ALL_CAM = False

if VISUALIZE:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    plt.ion()
    _, ax = plt.subplots(2, 2)
    cmap = cm.get_cmap('jet')


for row in trange(start_row, len(cadc_stats)):
    # print('\n------ Date %s, Sequence %d ------' % (cadc_stats.iloc[row, 0], cadc_stats.iloc[row, 1]))
    date = cadc_stats.iloc[row, 0]
    seq = cadc_stats.iloc[row, 1]
    n_frame = cadc_stats.iloc[row, 2]

    annotation_path = BASE + date + '/' + format(seq, '04') + "/3d_ann.json"
    # Load 3d annotations
    annotation = load_annotation(annotation_path)

    calib_path = BASE + date + '/' + "calib/"
    calib = load_calibration(calib_path)
    T_CAM_LIDAR = []
    T_IMG_CAM = []
    for cam in range(8):
        # Projection matrix from lidar to camera (extrinsics)
        T_CAM_LIDAR.append(np.linalg.inv(np.array(calib['extrinsics']['T_LIDAR_CAM0' + str(cam)]))[:3, :4])
        # Projection matrix from camera to image (intrinsics)
        T_IMG_CAM.append(np.array(calib['CAM0' + str(cam)]['camera_matrix']['data']).reshape(3, 3))

    for frame in trange(0, n_frame):
        # print('Loading lidar %d of %d (%.0f%%)' % (frame + 1, n_frame, (frame + 1) / n_frame * 100))

        cam_range = (1, 8) if ALL_CAM else (0, 1)
        for cam in range(*cam_range):
            # print('\tGenerating depth map of camera %d' % (cam))
            if generate_Raw:
                lidar_raw_path = BASE + date + '/' + format(seq, '04') + "/labeled/lidar_points/data/" + \
                                 format(frame, '010') + ".bin"
                lidar_raw = np.fromfile(lidar_raw_path, dtype=np.float32).reshape((-1, 4))
                # Project points onto image
                lidar_raw_projected = Lidar2Cam(lidar_raw, T_IMG_CAM[cam], T_CAM_LIDAR[cam])
                # Crop points to image view field
                lidar_raw_cropped = CropPoints(lidar_raw_projected)
                # Generate depth map (H, W) from points (N, 3)
                depth_raw = GenerateDepth(lidar_raw_cropped)
                # Save depth map to PIL
                depth_raw_path = BASE_mod + date + '/' + format(seq, '04') + "/labeled/image_0" + str(cam) + \
                                        "/depth/" + format(frame, '010') + ".png"
                if not (os.path.exists(depth_raw_path[:-14])):
                    os.makedirs(depth_raw_path[:-14])
                depth_raw_PIL = Image.fromarray(np.clip(depth_raw * 256., 0, 65535)).convert('I')
                depth_raw_PIL.save(depth_raw_path, mode='I;16')

            if generate_DROR:
                lidar_dror_path = BASE_mod + date + '/' + format(seq, '04') + "/labeled/lidar_points/lidar_dror/" + \
                                  format(frame, '010') + ".npy"
                lidar_dror = np.load(lidar_dror_path).reshape((-1, 4))
                # Project points onto image
                lidar_dror_projected = Lidar2Cam(lidar_dror, T_IMG_CAM[cam], T_CAM_LIDAR[cam])
                # Crop points to image view field
                lidar_dror_cropped = CropPoints(lidar_dror_projected)
                # Generate depth map (H, W) from points (N, 3)
                depth_dror = GenerateDepth(lidar_dror_cropped)
                # Save depth map to PIL
                depth_dror_path = BASE_mod + date + '/' + format(seq, '04') + "/labeled/image_0" + str(cam) + \
                                        "/depth_dror/" + format(frame, '010') + ".png"
                if not (os.path.exists(depth_dror_path[:-14])):
                    os.makedirs(depth_dror_path[:-14])
                depth_dror_PIL = Image.fromarray(np.clip(depth_dror * 256., 0, 65535)).convert('I')
                depth_dror_PIL.save(depth_dror_path, mode='I;16')

            if generate_aggregated_3:
                lidar_aggregate_path = BASE_mod + date + '/' + format(seq, '04') + \
                                         "/labeled/lidar_points/lidar_aggregated_3/" + \
                                         format(frame, '010') + ".npy"
                lidar_aggregate = np.load(lidar_aggregate_path).reshape((-1, 3))
                # Project points onto image
                lidar_aggregate_projected = Lidar2Cam(lidar_aggregate, T_IMG_CAM[cam], T_CAM_LIDAR[cam])
                # Crop points to image view field
                lidar_aggregate_cropped = CropPoints(lidar_aggregate_projected)
                # Generate depth map (H, W) from points (N, 3)
                depth_aggregate = GenerateDepth(lidar_aggregate_cropped)
                # Save depth map to PIL
                depth_aggregate_path = BASE_mod + date + '/' + format(seq, '04') + "/labeled/image_0" + str(cam) + \
                                        "/depth_aggregated_3/" + format(frame, '010') + ".png"
                if not (os.path.exists(depth_aggregate_path[:-14])):
                    os.makedirs(depth_aggregate_path[:-14])
                depth_aggregate_PIL = Image.fromarray(np.clip(depth_aggregate * 256., 0, 65535)).convert('I')
                depth_aggregate_PIL.save(depth_aggregate_path, mode='I;16')

            if generate_aggregated:
                lidar_aggregate_path = BASE_mod + date + '/' + format(seq, '04') + \
                                         "/labeled/lidar_points/lidar_aggregated/" + \
                                         format(frame, '010') + ".npy"
                lidar_aggregate = np.load(lidar_aggregate_path).reshape((-1, 3))
                # Project points onto image
                lidar_aggregate_projected = Lidar2Cam(lidar_aggregate, T_IMG_CAM[cam], T_CAM_LIDAR[cam])
                # Crop points to image view field
                lidar_aggregate_cropped = CropPoints(lidar_aggregate_projected)
                # Generate depth map (H, W) from points (N, 3)
                depth_aggregate = GenerateDepth(lidar_aggregate_cropped)
                # Save depth map to PIL
                depth_aggregate_path = BASE_mod + date + '/' + format(seq, '04') + "/labeled/image_0" + str(cam) + \
                                        "/depth_aggregated/" + format(frame, '010') + ".png"
                if not (os.path.exists(depth_aggregate_path[:-14])):
                    os.makedirs(depth_aggregate_path[:-14])
                depth_aggregate_PIL = Image.fromarray(np.clip(depth_aggregate * 256., 0, 65535)).convert('I')
                depth_aggregate_PIL.save(depth_aggregate_path, mode='I;16')

            if generate_HPR_ProjectedKNN:
                lidar_ProjectedKNN_path = BASE_mod + date + '/' + format(seq, '04') + \
                                          "/labeled/lidar_points/lidar_HPR_ProjectedKNN/" + \
                                          format(frame, '010') + ".npy"
                lidar_ProjectedKNN = np.load(lidar_ProjectedKNN_path).reshape((-1, 3))
                # Project points onto image
                lidar_ProjectedKNN_projected = Lidar2Cam(lidar_ProjectedKNN, T_IMG_CAM[cam], T_CAM_LIDAR[cam])
                # Crop points to image view field
                # lidar_ProjectedKNN_cropped = CropPoints(lidar_ProjectedKNN_projected)
                # Generate depth map (H, W) from points (N, 3)
                depth_ProjectedKNN = GenerateDepth(lidar_ProjectedKNN_projected)
                # Save depth map to PIL
                depth_ProjectedKNN_path = BASE_mod + date + '/' + format(seq, '04') + "/labeled/image_0" + str(cam) + \
                                          "/depth_HPR_ProjectedKNN/" + format(frame, '010') + ".png"
                if not (os.path.exists(depth_ProjectedKNN_path[:-14])):
                    os.makedirs(depth_ProjectedKNN_path[:-14])
                depth_ProjectedKNN_PIL = Image.fromarray(np.clip(depth_ProjectedKNN * 256., 0, 65535)).convert('I')
                depth_ProjectedKNN_PIL.save(depth_ProjectedKNN_path, mode='I;16')

            if generate_HPR_ProjectedKNN_99:
                lidar_ProjectedKNN_path = BASE_mod + date + '/' + format(seq, '04') + \
                                          "/labeled/lidar_points/lidar_HPR_ProjectedKNN_0.99/" + \
                                          format(frame, '010') + ".npy"
                lidar_ProjectedKNN = np.load(lidar_ProjectedKNN_path).reshape((-1, 3))
                # Project points onto image
                lidar_ProjectedKNN_projected = Lidar2Cam(lidar_ProjectedKNN, T_IMG_CAM[cam], T_CAM_LIDAR[cam])
                # Crop points to image view field
                # lidar_ProjectedKNN_cropped = CropPoints(lidar_ProjectedKNN_projected)
                # Generate depth map (H, W) from points (N, 3)
                depth_ProjectedKNN = GenerateDepth(lidar_ProjectedKNN_projected)
                # Save depth map to PIL
                depth_ProjectedKNN_path = BASE_mod + date + '/' + format(seq, '04') + "/labeled/image_0" + str(
                    cam) + \
                                          "/depth_HPR_ProjectedKNN_0.99/" + format(frame, '010') + ".png"
                if not (os.path.exists(depth_ProjectedKNN_path[:-14])):
                    os.makedirs(depth_ProjectedKNN_path[:-14])
                depth_ProjectedKNN_PIL = Image.fromarray(np.clip(depth_ProjectedKNN * 256., 0, 65535)).convert('I')
                depth_ProjectedKNN_PIL.save(depth_ProjectedKNN_path, mode='I;16')

            if generate_HPR_ConvexHull:
                lidar_ConvexHull_path = BASE_mod + date + '/' + format(seq, '04') + \
                                         "/labeled/lidar_points/lidar_HPR_ConvexHull/" + \
                                         format(frame, '010') + ".npy"
                lidar_ConvexHull = np.load(lidar_ConvexHull_path).reshape((-1, 3))
                # Project points onto image
                lidar_ConvexHull_projected = Lidar2Cam(lidar_ConvexHull, T_IMG_CAM[cam], T_CAM_LIDAR[cam])
                # Crop points to image view field
                # lidar_ConvexHull_cropped = CropPoints(lidar_ConvexHull_projected)
                # Generate depth map (H, W) from points (N, 3)
                depth_ConvexHull = GenerateDepth(lidar_ConvexHull_projected)
                # Save depth map to PIL
                depth_ConvexHull_path = BASE_mod + date + '/' + format(seq, '04') + "/labeled/image_0" + str(cam) + \
                                        "/depth_HPR_ConvexHull/" + format(frame, '010') + ".png"
                if not (os.path.exists(depth_ConvexHull_path[:-14])):
                    os.makedirs(depth_ConvexHull_path[:-14])
                depth_ConvexHull_PIL = Image.fromarray(np.clip(depth_ConvexHull * 256., 0, 65535)).convert('I')
                depth_ConvexHull_PIL.save(depth_ConvexHull_path, mode='I;16')

            # if VISUALIZE:
            #     # visualization
            #     img_path = BASE + date + '/' + format(seq, '04') + "/labeled" + "/image_0" + str(cam) + \
            #                "/data/" + format(frame, '010') + ".png"
            #     img = plt.imread(img_path)
            #     depth_raw_path = BASE_mod + date + '/' + format(seq, '04') + "/labeled" + "/image_0" + str(cam) + \
            #                      "/depth/" + format(frame, '010') + ".png"
            #     depth_raw = np.array(Image.open(depth_raw_path), dtype=int).astype(np.float32) / 256.
            #     depth_dror_path = BASE_mod + date + '/' + format(seq, '04') + "/labeled" + "/image_0" + str(cam) + \
            #                       "/depth_dror/" + format(frame, '010') + ".png"
            #     depth_dror = np.array(Image.open(depth_dror_path), dtype=int).astype(np.float32) / 256.
            #
            #     ax[0, 0].clear()
            #     ax[0, 0].imshow(img)
            #     ax[0, 0].set_title('RGB image')
            #     ax[0, 1].clear()
            #     ax[0, 1].imshow(img)
            #     ax[0, 1].scatter(np.nonzero(depth_raw)[1], np.nonzero(depth_raw)[0],
            #                      c=cmap(np.clip(depth_raw[np.nonzero(depth_raw)], 0, 80) / 80), s=0.001)
            #     ax[0, 1].set_title('Raw Depth map')
            #     ax[1, 0].clear()
            #     ax[1, 0].imshow(img)
            #     ax[1, 0].scatter(np.nonzero(depth_dror)[1], np.nonzero(depth_dror)[0],
            #                      c=cmap(np.clip(depth_dror[np.nonzero(depth_dror)], 0, 80) / 80), s=0.001)
            #     ax[1, 0].set_title('DROR Depth map')
            #     ax[1, 1].clear()
            #     ax[1, 1].imshow(img)
            #     ax[1, 1].scatter(np.nonzero(depth_aggregate)[1], np.nonzero(depth_aggregate)[0],
            #                      c=cmap(np.clip(depth_aggregate[np.nonzero(depth_aggregate)], 0, 80) / 80), s=0.001)
            #     ax[1, 1].set_title('Aggregated Depth map')
            #
            #     plt.pause(0.1)
            #     plt.show()
