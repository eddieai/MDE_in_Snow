import numpy as np
from numpy.linalg import inv
from scipy.spatial.transform import Rotation as R
import open3d as o3d
from cadc_utils import load_annotation


def find_cuboid_uuid(lidar, annotation, margin=0.2, show3D=False):
    print('Lidar points:', len(lidar))

    # Add each cuboid to image
    for cuboid in annotation['cuboids']:
        if cuboid['points_count'] < 200:
            continue
        print(cuboid['uuid'])

        # if cuboid['label'] in ['Car', 'Truck', 'Bus', 'Bicycle', 'Pedestrian', 'Pedestrian with Object', 'Animal', 'Garbage Container on Wheels']:
        T_Lidar_Cuboid = R.from_euler('z', cuboid['yaw'], degrees=False).as_matrix()  # rotate the identity matrix
        lidar_rotate = np.dot(lidar[:, :3], inv(T_Lidar_Cuboid).T)
        center_rotate = np.dot(np.array([cuboid['position']['x'], cuboid['position']['y'], cuboid['position']['z']]), inv(T_Lidar_Cuboid).T)

        width = cuboid['dimensions']['x'] + margin
        length = cuboid['dimensions']['y'] + margin
        height = cuboid['dimensions']['z'] + margin

        threshold_min = np.array([center_rotate[0] - length/2, center_rotate[1] - width/2, center_rotate[2] - height/2]).reshape(1,3)
        threshold_max = np.array([center_rotate[0] + length/2, center_rotate[1] + width/2, center_rotate[2] + height/2]).reshape(1,3)

        cuboid_points_idx = np.argwhere(np.all(lidar_rotate >= threshold_min, axis=1) & np.all(lidar_rotate <= threshold_max, axis=1)).squeeze()

        if show3D:
            lidar_3d = o3d.geometry.PointCloud()
            lidar_3d.points = o3d.utility.Vector3dVector(lidar[:, :3])
            lidar_3d.colors = o3d.utility.Vector3dVector(np.ones((len(lidar), 3)) * 0.5)
            np.asarray(lidar_3d.colors)[cuboid_points_idx] = [1, 0, 0]
            o3d.visualization.draw_geometries([lidar_3d])


def trace_cuboid(uuid, lidars, annotations, margin=0.2, show3D=False):
    for frame, lidar in enumerate(lidars):
        print('frame ', frame)
        annotation = annotations[frame]

        # Add each cuboid to image
        for cuboid in annotation['cuboids']:
            if cuboid['uuid'] == uuid:
                # if cuboid['label'] in ['Car', 'Truck', 'Bus', 'Bicycle', 'Pedestrian', 'Pedestrian with Object', 'Animal', 'Garbage Container on Wheels']:
                T_Lidar_Cuboid = R.from_euler('z', cuboid['yaw'], degrees=False).as_matrix()  # rotate the identity matrix
                lidar_rotate = np.dot(lidar[:, :3], inv(T_Lidar_Cuboid).T)
                center_rotate = np.dot(np.array([cuboid['position']['x'], cuboid['position']['y'], cuboid['position']['z']]), inv(T_Lidar_Cuboid).T)

                width = cuboid['dimensions']['x'] + margin
                length = cuboid['dimensions']['y'] + margin
                height = cuboid['dimensions']['z'] + margin

                threshold_min = np.array([center_rotate[0] - length/2, center_rotate[1] - width/2, center_rotate[2] - height/2]).reshape(1,3)
                threshold_max = np.array([center_rotate[0] + length/2, center_rotate[1] + width/2, center_rotate[2] + height/2]).reshape(1,3)

                cuboid_points_idx = np.argwhere(np.all(lidar_rotate >= threshold_min, axis=1) & np.all(lidar_rotate <= threshold_max, axis=1)).squeeze()

                if show3D:
                    lidar_3d = o3d.geometry.PointCloud()
                    lidar_3d.points = o3d.utility.Vector3dVector(lidar[:, :3])
                    lidar_3d.colors = o3d.utility.Vector3dVector(np.ones((len(lidar), 3)) * 0.5)
                    np.asarray(lidar_3d.colors)[cuboid_points_idx] = [1, 0, 0]
                    o3d.visualization.draw_geometries([lidar_3d])


if __name__ == '__main__':
    BASE = '/home/datasets/CADC/cadcd/'
    BASE_mod = '/home/datasets_mod/CADC/cadcd/'
    date = '2019_02_27'
    seq = 20
    frame = 5
    lidar_type = 'dror'  # raw, dror, aggregated

    annotations_path = BASE + date + '/' + format(seq, '04') + "/3d_ann.json"
    # Load 3d annotations
    annotations = load_annotation(annotations_path)

    lidars_dror = []
    for i in range(len(annotations)):
        print('Loading lidar %d of %d (%.0f%%)' % (i + 1, len(annotations), (i + 1) / len(annotations) * 100))
        lidar_dror_path = BASE_mod + date + '/' + format(seq, '04') + "/labeled/lidar_points/lidar_dror/" + \
                          format(i, '010') + ".npy"
        lidar_dror = np.load(lidar_dror_path).reshape((-1, 4))
        lidars_dror.append(lidar_dror)

    # find_cuboid_uuid(lidars_dror[frame], annotations[frame], show3D=True)

    uuid = '339d1ba2-afb2-4813-8121-0c65ab8dc658'
    trace_cuboid(uuid, lidars_dror, annotations, show3D=True)
