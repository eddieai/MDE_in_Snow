from tqdm import *
import numpy as np
from numpy.linalg import inv
from numpy.linalg import multi_dot
import open3d as o3d
from cadc_utils import load_novatel_data, load_annotation, load_calibration, convert_novatel_to_pose
from cadc_utils import Lidar2Cam, CropPoints
from filter_lidar_dror import DROR_filter
from filter_lidar_dynamic_objects import lidar_mobile_objects_filter
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def lidar_aggregate(ref_frame, lidars_no_mobile, cuboid_aggregated, poses, show_3D=False):
    rotation_ref, translation_ref = poses[ref_frame]

    lidar_aggregated = cuboid_aggregated

    if show_3D:
        cmap = plt.get_cmap('YlGnBu')
        norm = colors.Normalize(vmin=0, vmax=len(lidars_no_mobile))
        scalarMap = cm.ScalarMappable(norm=norm, cmap=cmap)
        lidar_colors = np.broadcast_to(np.array([1,0,0]), (len(cuboid_aggregated),3))

    for frame in range(len(lidars_no_mobile)):
        lidar_no_mobile = lidars_no_mobile[frame]

        rotation, translation = poses[frame]
        transformation = multi_dot([rotation_ref, translation_ref, inv(translation), inv(rotation)])[:3, :]

        lidar_transformed = np.dot(np.hstack((lidar_no_mobile[:, :3], np.ones((len(lidar_no_mobile), 1)))), transformation.T)
        lidar_aggregated = np.vstack((lidar_aggregated, lidar_transformed))

        if show_3D:
            lidar_color = np.broadcast_to(scalarMap.to_rgba(frame)[:3], (len(lidar_transformed), 3))
            lidar_colors = np.vstack((lidar_colors, lidar_color))

    if show_3D:
        ## Display aggregated point clouds
        lidar_aggregated_3d = o3d.geometry.PointCloud()
        lidar_aggregated_3d.points = o3d.utility.Vector3dVector(lidar_aggregated)
        lidar_aggregated_3d.colors = o3d.utility.Vector3dVector(lidar_colors)
        o3d.visualization.draw_geometries([lidar_aggregated_3d])

    return lidar_aggregated


# def lidar_aggregate_visualize(allcam_img, lidar_dror, lidar_aggregated, calib, show_cam0=True, show_allcam=False):
#     if show_allcam:
#         # Project to all 8 cameras
#         for cam in range(8):
#             # Projection matrix from lidar to camera (extrinsics)
#             T_CAM_LIDAR = inv(np.array(calib['extrinsics']['T_LIDAR_CAM0' + str(cam)]))[:3, :4]
#             # Projection matrix from camera to image (intrinsics)
#             T_IMG_CAM = np.array(calib['CAM0' + str(cam)]['camera_matrix']['data']).reshape(3, 3)
#
#             # Project points onto image
#             lidar_projected = Lidar2Cam(lidar_dror[:, :3], T_IMG_CAM, T_CAM_LIDAR)
#             lidar_cropped = CropPoints(lidar_projected)
#             lidar_aggregated_projected = Lidar2Cam(lidar_aggregated[:, :3], T_IMG_CAM, T_CAM_LIDAR)
#             lidar_aggregated_cropped = CropPoints(lidar_aggregated_projected)
#
#             ax_allcam[0, cam].clear()
#             ax_allcam[0, cam].imshow(allcam_img[cam])
#             ax_allcam[0, cam].set_xticks([])
#             ax_allcam[0, cam].set_yticks([])
#             ax_allcam[1, cam].clear()
#             ax_allcam[1, cam].imshow(allcam_img[cam])
#             ax_allcam[1, cam].scatter(lidar_cropped[:, 0], lidar_cropped[:, 1], s=0.005,
#                                       c=cmap(np.clip(lidar_cropped[:, 2], 0, 80) / 80))
#             ax_allcam[1, cam].set_xlim(0, allcam_img[cam].shape[1])
#             ax_allcam[1, cam].set_xticks([])
#             ax_allcam[1, cam].set_yticks([])
#             ax_allcam[2, cam].clear()
#             ax_allcam[2, cam].imshow(allcam_img[cam])
#             ax_allcam[2, cam].scatter(lidar_aggregated_cropped[:, 0], lidar_aggregated_cropped[:, 1], s=0.005,
#                                       c=cmap(np.clip(lidar_aggregated_cropped[:, 2], 0, 80) / 80))
#             ax_allcam[2, cam].set_xlim(0, allcam_img[cam].shape[1])
#             ax_allcam[2, cam].set_xticks([])
#             ax_allcam[2, cam].set_yticks([])
#
#     if show_cam0:
#         # Project to camera 0
#         # Projection matrix from lidar to camera (extrinsics)
#         T_CAM_LIDAR = inv(np.array(calib['extrinsics']['T_LIDAR_CAM00']))[:3, :4]
#         # Projection matrix from camera to image (intrinsics)
#         T_IMG_CAM = np.array(calib['CAM00']['camera_matrix']['data']).reshape(3, 3)
#
#         # Project points onto image
#         lidar_projected = Lidar2Cam(lidar_dror[:, :3], T_IMG_CAM, T_CAM_LIDAR)
#         lidar_cropped = CropPoints(lidar_projected)
#         lidar_aggregated_projected = Lidar2Cam(lidar_aggregated[:, :3], T_IMG_CAM, T_CAM_LIDAR)
#         lidar_aggregated_cropped = CropPoints(lidar_aggregated_projected)
#
#         ax_cam0[0].clear()
#         ax_cam0[0].imshow(allcam_img[0])
#         ax_cam0[1].clear()
#         ax_cam0[1].imshow(allcam_img[0])
#         ax_cam0[1].scatter(lidar_cropped[:, 0], lidar_cropped[:, 1], s=0.01,
#                            c=cmap(np.clip(lidar_cropped[:, 2], 0, 80) / 80))
#         ax_cam0[1].set_xlim(0, allcam_img[0].shape[1])
#         ax_cam0[2].clear()
#         ax_cam0[2].imshow(allcam_img[0])
#         ax_cam0[2].scatter(lidar_aggregated_cropped[:, 0], lidar_aggregated_cropped[:, 1], s=0.01,
#                            c=cmap(np.clip(lidar_aggregated_cropped[:, 2], 0, 80) / 80))
#         ax_cam0[2].set_xlim(0, allcam_img[0].shape[1])


if __name__ == '__main__':
    BASE = "/home/datasets/CADC/cadcd/"
    BASE_mod = "/home/datasets_mod/CADC/cadcd/"
    date = '2018_03_06'
    seq = 10  # 10
    start_frame = 15  # 40
    n_aggregate = 11  # n_aggregate % 2 == 1
    assert n_aggregate % 2 == 1

    show_3D = True
    show_cam0 = False
    show_allcam = False

    # if show_cam0 or show_allcam:
    #     cmap = cm.get_cmap('jet')
    #     if show_cam0:
    #         _, ax_cam0 = plt.subplots(1, 3)
    #     if show_allcam:
    #         _, ax_allcam = plt.subplots(3, 8)

    # Load 3d annotations
    annotations_path = BASE + date + '/' + format(seq, '04') + "/3d_ann.json"
    annotations = load_annotation(annotations_path)

    # Load novatel IMU data
    novatel_path = BASE + date + '/' + format(seq, '04') + "/labeled/novatel/data/"
    novatel = load_novatel_data(novatel_path)
    poses = convert_novatel_to_pose(novatel)

    # calib_path = BASE + date + '/' + "calib/"
    # calib = load_calibration(calib_path)

    lidars_no_mobile = []
    for frame in range(len(annotations)):
        lidar_no_mobile_path = BASE_mod + date + '/' + format(seq, '04') + "/labeled/lidar_points/lidar_no_mobile/" + \
                               format(frame, '010') + ".npy"
        lidar_no_mobile = np.load(lidar_no_mobile_path).reshape((-1, 3))
        lidars_no_mobile.append(lidar_no_mobile)

    print('\n')
    # plt.ion()

    for frame in trange(start_frame, len(annotations)):
        # print('Aggregating frame %d of %d (%.0f%%)' % (ref_frame+1, len(annotation), (ref_frame+1)/len(annotation)*100))
        cuboid_aggregated_path = BASE_mod + date + '/' + format(seq, '04') + "/labeled/lidar_points/cuboid_aggregated/" + \
                               format(frame, '010') + ".npy"
        cuboid_aggregated = np.load(cuboid_aggregated_path).reshape((-1, 3))

        min_frame = max(0, frame - (n_aggregate - 1) // 2)
        max_frame = min(len(lidars_no_mobile), frame + (n_aggregate - 1) // 2 + 1)
        ref_frame = min(frame, (n_aggregate - 1) // 2)

        lidar_aggregated = lidar_aggregate(ref_frame, lidars_no_mobile[min_frame:max_frame], cuboid_aggregated, poses[min_frame:max_frame], show_3D=show_3D)

        # if show_cam0 or show_allcam:
        #     allcam_img = []
        #     for cam in range(8):
        #         img_path = BASE + date + '/' + format(seq, '04') + "/labeled/image_0" + str(cam) + "/data/" + \
        #                    format(ref_frame,'010') + ".png"
        #         img = plt.imread(img_path)
        #         allcam_img.append(img)
        #
        #     lidar_aggregate_visualize(allcam_img, lidar_dror, lidar_aggregated, calib,
        #                               show_cam0=show_cam0, show_allcam=show_allcam)
        #
        #     plt.pause(0.1)
        #     plt.show()
