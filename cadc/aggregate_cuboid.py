from tqdm import *
import numpy as np
from numpy.linalg import inv
from scipy.spatial.transform import Rotation as R
import open3d as o3d
from cadc_utils import load_annotation


def cuboid_aggregate(ref_frame, lidars_dror, annotations, show3D=False):
    annotation_ref = annotations[ref_frame]

    agg_cuboids = {}

    for frame in range(len(lidars_dror)):
        lidar = lidars_dror[frame][:, :3]
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

    cuboid_aggregated = np.zeros((0,3))

    for cuboid in annotation_ref['cuboids']:
        if not cuboid['stationary']:
            uuid = cuboid['uuid']
            yaw = cuboid['yaw']
            center = np.array([cuboid['position']['x'], cuboid['position']['y'], cuboid['position']['z']])

            transformation_cuboid = np.eye(4)
            transformation_cuboid[:3, :3] = R.from_euler('z', yaw, degrees=False).as_matrix()
            transformation_cuboid[:3, 3] = center

            agg_cuboid = agg_cuboids[uuid]
            agg_cuboid_transformed = np.dot(np.c_[agg_cuboid, np.ones(len(agg_cuboid))], transformation_cuboid.T)[:, :3]

            cuboid_aggregated = np.vstack((cuboid_aggregated, agg_cuboid_transformed))

    if show3D:
        cuboid_aggregated_3d = o3d.geometry.PointCloud()
        cuboid_aggregated_3d.points = o3d.utility.Vector3dVector(cuboid_aggregated)
        o3d.visualization.draw_geometries([cuboid_aggregated_3d])

    return cuboid_aggregated


if __name__ == '__main__':
    BASE = '/home/datasets/CADC/cadcd/'
    BASE_mod = '/home/datasets_mod/CADC/cadcd/'
    date = '2019_02_27'     # '2019_02_27'
    seq = 27     # 20
    start_frame = 0
    n_aggregate = 11  # n_aggregate % 2 == 1
    assert n_aggregate % 2 == 1

    annotations_path = BASE + date + '/' + format(seq, '04') + "/3d_ann.json"
    annotations = load_annotation(annotations_path)

    lidars_dror = []
    for frame in trange(len(annotations)):
        # print('Loading lidar %d of %d (%.0f%%)' % (i + 1, len(annotations), (i + 1) / len(annotations) * 100))
        lidar_dror_path = BASE_mod + date + '/' + format(seq, '04') + "/labeled/lidar_points/lidar_dror/" + \
                          format(frame, '010') + ".npy"
        lidar_dror = np.load(lidar_dror_path).reshape((-1, 4))
        lidars_dror.append(lidar_dror)

    for frame in trange(start_frame, len(annotations)):
        min_frame = max(0, frame - (n_aggregate - 1) // 2)
        max_frame = min(len(lidars_dror), frame + (n_aggregate - 1) // 2 + 1)
        ref_frame = min(frame, (n_aggregate - 1) // 2)

        cuboid_aggregated = cuboid_aggregate(ref_frame, lidars_dror[min_frame:max_frame], annotations[min_frame:max_frame], show3D=True)
