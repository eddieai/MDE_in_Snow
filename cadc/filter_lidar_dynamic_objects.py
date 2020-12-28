from tqdm import *
import numpy as np
from numpy.linalg import inv
from scipy.spatial.transform import Rotation as R
import open3d as o3d
from cadc_utils import load_annotation


def lidar_mobile_objects_filter(lidar, annotation, show3D=False):
    # print('Lidar points:', len(lidar))
    deleted = []

    # Add each cuboid to image
    for cuboid in tqdm(annotation['cuboids']):
        if not cuboid['stationary']:
            yaw = cuboid['yaw']
            center = np.array([cuboid['position']['x'], cuboid['position']['y'], cuboid['position']['z']])

            transformation_cuboid = np.eye(4)
            transformation_cuboid[:3, :3] = R.from_euler('z', yaw, degrees=False).as_matrix()
            transformation_cuboid[:3, 3] = center

            lidar_transformed = np.dot(np.c_[lidar, np.ones(len(lidar))], inv(transformation_cuboid).T)[:, :3]

            width = cuboid['dimensions']['x']
            length = cuboid['dimensions']['y']
            height = cuboid['dimensions']['z']

            threshold_min = np.array([- length / 2, - width / 2, - height / 2]).reshape(1, 3)
            threshold_max = np.array([length / 2, width / 2, height / 2]).reshape(1, 3)
            cuboid_points_idx = np.nonzero(np.all((lidar_transformed >= threshold_min) &
                                                   (lidar_transformed <= threshold_max), axis=1))[0].tolist()

            deleted += cuboid_points_idx

    filtered = list(set(range(len(lidar))) - set(deleted))
    # print('Lidar points after removing points of mobile objects:', len(filtered))

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
    seq = 1
    frame = 5
    lidar_type = 'dror'  # raw, dror, aggregated

    if lidar_type == 'raw':
        lidar_path = BASE + date + '/' + format(seq, '04') + \
                     "/labeled/lidar_points/data/" + format(frame, '010') + ".bin"
        lidar = np.fromfile(lidar_path, dtype=np.float32).reshape((-1, 4))[:, :3]
    elif lidar_type == 'dror':
        lidar_path = BASE_mod + date + '/' + format(seq, '04') + \
                     "/labeled/lidar_points/lidar_dror/" + format(frame, '010') + ".npy"
        lidar = np.load(lidar_path).reshape((-1, 4))[:, :3]
    elif lidar_type == 'aggregated':
        lidar_path = BASE_mod + date + '/' + format(seq, '04') + \
                     "/labeled/lidar_points/lidar_aggregated/" + format(frame, '010') + ".npy"
        lidar = np.load(lidar_path).reshape((-1, 3))

    annotations_path = BASE + date + '/' + format(seq, '04') + "/3d_ann.json"
    # Load 3d annotations
    annotation = load_annotation(annotations_path)[frame]

    filtered = lidar_mobile_objects_filter(lidar, annotation, show3D=True)

    lidar_filtered_3d = o3d.geometry.PointCloud()
    lidar_filtered_3d.points = o3d.utility.Vector3dVector(lidar[filtered])
    o3d.visualization.draw_geometries([lidar_filtered_3d])
