from tqdm import *
import numpy as np
from numpy.linalg import inv
from scipy.spatial.transform import Rotation as R
import open3d as o3d
from cadc_utils import load_annotation


def cuboid_aggregate(lidars, annotations):
    agg_cuboids = {}

    for frame in trange(len(annotations)):
        lidar = lidars[frame][:, :3]
        # print('Aggregating cuboids of frame %d' % (frame))
        annotation = annotations[frame]

        # Add each cuboid to image
        for cuboid in annotation['cuboids']:
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

                threshold_min = np.array([- length/2, - width/2, - height/2]).reshape(1,3)
                threshold_max = np.array([length/2, width/2, height/2]).reshape(1,3)
                cuboid_points = lidar_transformed[np.all((lidar_transformed >= threshold_min) &
                                                         (lidar_transformed <= threshold_max), axis=1)]

                if cuboid['uuid'] in agg_cuboids:
                    agg_cuboids[cuboid['uuid']] = np.vstack((agg_cuboids[cuboid['uuid']], cuboid_points))
                else:
                    agg_cuboids[cuboid['uuid']] = cuboid_points

    return agg_cuboids


def agg_cuboid_to_frame(agg_cuboids, annotation, show3D=False):
    point_cloud = np.zeros((0,3))

    for cuboid in annotation['cuboids']:
        if not cuboid['stationary']:
            uuid = cuboid['uuid']
            yaw = cuboid['yaw']
            center = np.array([cuboid['position']['x'], cuboid['position']['y'], cuboid['position']['z']])

            transformation_cuboid = np.eye(4)
            transformation_cuboid[:3, :3] = R.from_euler('z', yaw, degrees=False).as_matrix()
            transformation_cuboid[:3, 3] = center

            agg_cuboid = agg_cuboids[uuid]
            agg_cuboid_transformed = np.dot(np.c_[agg_cuboid, np.ones(len(agg_cuboid))], transformation_cuboid.T)[:, :3]

            point_cloud = np.vstack((point_cloud, agg_cuboid_transformed))

    if show3D:
        point_cloud_3d = o3d.geometry.PointCloud()
        point_cloud_3d.points = o3d.utility.Vector3dVector(point_cloud)
        o3d.visualization.draw_geometries([point_cloud_3d])

    return point_cloud


if __name__ == '__main__':
    BASE = '/home/datasets/CADC/cadcd/'
    BASE_mod = '/home/datasets_mod/CADC/cadcd/'
    date = '2019_02_27'     # '2019_02_27'
    seq = 20     # 20
    start_frame = 15

    annotations_path = BASE + date + '/' + format(seq, '04') + "/3d_ann.json"
    annotations = load_annotation(annotations_path)

    lidars_dror = []
    for frame in trange(len(annotations)):
        # print('Loading lidar %d of %d (%.0f%%)' % (i + 1, len(annotations), (i + 1) / len(annotations) * 100))
        lidar_dror_path = BASE_mod + date + '/' + format(seq, '04') + "/labeled/lidar_points/lidar_dror/" + \
                          format(frame, '010') + ".npy"
        lidar_dror = np.load(lidar_dror_path).reshape((-1, 4))
        lidars_dror.append(lidar_dror)

    # agg_cuboids = cuboid_aggregate(lidars_dror, annotations)
    agg_cuboids_path = BASE_mod + date + '/' + format(seq, '04') + \
                       "/labeled/lidar_points/cuboid_aggregated/agg_cuboids.npy"
    agg_cuboids = np.load(agg_cuboids_path, allow_pickle=True).item()

    for frame in trange(start_frame, len(annotations)):
        point_cloud = agg_cuboid_to_frame(agg_cuboids, annotations[frame], show3D=True)
