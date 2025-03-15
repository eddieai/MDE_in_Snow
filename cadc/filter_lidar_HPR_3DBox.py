from tqdm import *
import numpy as np
from numpy.linalg import inv
from scipy.spatial.qhull import ConvexHull
from matplotlib.path import Path
from scipy.spatial.transform import Rotation as R
import open3d as o3d
from cadc_utils import load_annotation, load_calibration, Lidar2Cam


def HPR_3DBox(lidar, annotation, calib, show3D=False):
    # Projection matrix from camera to image frame
    T_IMG_CAM = np.eye(4)
    T_IMG_CAM[0:3, 0:3] = np.array(calib['CAM00']['camera_matrix']['data']).reshape(-1, 3)
    T_IMG_CAM = T_IMG_CAM[0:3, 0:4]  # remove last row
    T_CAM_LIDAR = np.linalg.inv(np.array(calib['extrinsics']['T_LIDAR_CAM00']))

    deleted = []
    for cuboid in annotation['cuboids']:
        if (not cuboid['stationary']) and (cuboid['dimensions']['x']>0):
            # Visualize points in frustum
            # Project 3D cuboid to 2D polygon
            yaw = cuboid['yaw']
            center = np.array([cuboid['position']['x'], cuboid['position']['y'], cuboid['position']['z']])
            transformation_cuboid = np.eye(4)
            transformation_cuboid[:3, :3] = R.from_euler('z', yaw, degrees=False).as_matrix()
            transformation_cuboid[:3, 3] = center

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
            t_lidar_cuboid = np.dot(cuboid_vertices, transformation_cuboid.T)
            t_cam_cuboid = np.dot(t_lidar_cuboid, T_CAM_LIDAR.T)
            t_img_cuboid = np.dot(t_cam_cuboid, T_IMG_CAM.T)
            cuboid_projected_vertices = np.stack([t_img_cuboid[:, 0] / t_img_cuboid[:, 2],
                                                  t_img_cuboid[:, 1] / t_img_cuboid[:, 2],
                                                  t_img_cuboid[:, 2]], axis=-1)

            if np.sum(
                    np.all(cuboid_projected_vertices > np.array([0, 0, 0]).reshape(1,3), axis=1) &
                    np.all(cuboid_projected_vertices < np.array([1280, 1024, np.inf]).reshape(1,3), axis=1)
            ) < 4:
                continue

            hull = ConvexHull(cuboid_projected_vertices[:, :2])
            polygon = Path(cuboid_projected_vertices[hull.vertices][:, :2])

            lidar_projected = Lidar2Cam(lidar, T_IMG_CAM, T_CAM_LIDAR)
            inside_polygon_idx = polygon.contains_points(lidar_projected[:, :2])
            behind_cuboid = lidar_projected[:, 2] > np.max(cuboid_projected_vertices[:, 2])

            deleted += np.nonzero(np.logical_and(inside_polygon_idx, behind_cuboid))[0].tolist()

    filtered = list(set(range(len(lidar))) - set(deleted))

    if show3D:
        lidar_3d = o3d.geometry.PointCloud()
        lidar_3d.points = o3d.utility.Vector3dVector(lidar)
        lidar_3d.colors = o3d.utility.Vector3dVector(np.ones((len(lidar), 3)) * 0.5)
        np.asarray(lidar_3d.colors)[list(set(deleted))] = [1, 0, 0]
        o3d.visualization.draw_geometries([lidar_3d])

    return filtered


if __name__ == '__main__':
    BASE = '/home/datasets/CADC/cadcd/'
    BASE_mod = '/home/datasets_mod/CADC/cadcd/'
    date = '2018_03_06'
    seq = 16
    frame = 38
    lidar_type = 'aggregated'  # raw, dror, aggregated

    annotations_path =  BASE + date + '/' + format(seq, '04') + "/3d_ann.json"
    annotation = load_annotation(annotations_path)[frame]

    calib_path = BASE + date + '/' + "calib"
    calib = load_calibration(calib_path)

    if lidar_type == 'raw':
        lidar_path = BASE + date + '/' + format(seq, '04') + \
                     "/labeled/lidar_points/data/" + format(frame, '010') + ".bin"
        lidar = np.fromfile(lidar_path, dtype= np.float32).reshape((-1, 4))[:,:3]
    elif lidar_type == 'dror':
        lidar_path = BASE_mod + date + '/' + format(seq, '04') + \
                     "/labeled/lidar_points/lidar_dror/" + format(frame, '010') + ".npy"
        lidar = np.load(lidar_path).reshape((-1, 4))[:,:3]
    elif lidar_type == 'aggregated':
        lidar_path = BASE_mod + date + '/' + format(seq, '04') + \
                     "/labeled/lidar_points/lidar_aggregated/" + format(frame, '010') + ".npy"
        lidar = np.load(lidar_path).reshape((-1, 3))

    filtered = HPR_3DBox(lidar, annotation, calib, show3D=True)

    lidar_filtered_3d = o3d.geometry.PointCloud()
    lidar_filtered_3d.points = o3d.utility.Vector3dVector(lidar[filtered])
    o3d.visualization.draw_geometries([lidar_filtered_3d])
