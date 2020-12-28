import numpy as np
import open3d as o3d


def DROR_filter(lidar, alpha=0.2, beta=3, k_min=3, SR_min=0.04, range_x=None, range_y=None, range_z=None, show_3D=False):
    print('Lidar points:', len(lidar))

    lidar_3d = o3d.geometry.PointCloud()
    lidar_3d.points = o3d.utility.Vector3dVector(lidar[:,:3])
    # lidar_3d.colors = o3d.utility.Vector3dVector(list(zip(lidar[:,3]/lidar[:,3].max(), np.zeros(len(lidar)), np.zeros(len(lidar)))))
    if show_3D:
        lidar_3d.colors = o3d.utility.Vector3dVector(np.ones((len(lidar), 3)) * 0.5)
        o3d.visualization.draw_geometries([lidar_3d])
        lidar_copy_3d = lidar_3d
        lidar_copy_3d.colors = o3d.utility.Vector3dVector(np.ones((len(lidar), 3)) * 0.5)

    lidar_kdtree = o3d.geometry.KDTreeFlann(lidar_3d)
    filtered = []

    for i in range(len(lidar)):
        x = lidar_3d.points[i][0]
        y = lidar_3d.points[i][1]
        z = lidar_3d.points[i][2]

        if (range_x is not None) and not (range_x[0] <= x <= range_x[1]):
            filtered.append(i)
            continue
        if (range_y is not None) and not (range_y[0] <= y <= range_y[1]):
            filtered.append(i)
            continue
        if (range_z is not None) and not (range_z[0] <= z <= range_z[1]):
            filtered.append(i)
            continue

        r = np.sqrt(x**2 + y**2)
        SR = np.max((SR_min, beta * r * alpha * np.pi / 180))
        [k, _, _] = lidar_kdtree.search_radius_vector_3d(lidar_3d.points[i], SR)

        if k >= k_min:
            filtered.append(i)
        elif show_3D:
            np.asarray(lidar_copy_3d.colors)[i] = [1, 0, 0]

    print('Lidar points after DROR filter:', len(filtered))

    if show_3D:
        o3d.visualization.draw_geometries([lidar_copy_3d])

    return filtered


if __name__ == '__main__':
    BASE = '/home/datasets/CADC/cadcd/'
    date = '2018_03_06'
    seq = 6
    frame = 5

    lidar_path = BASE + date + '/' + format(seq, '04') + "/labeled/lidar_points/data/" + format(frame, '010') + ".bin"
    scan_data = np.fromfile(lidar_path, dtype= np.float32)
    # scan_data is a single row of all the lidar values, 2D array where each row contains a point [x, y, z, intensity]
    lidar = scan_data.reshape((-1, 4))[:,:3]

    filtered = DROR_filter(lidar, alpha=0.2, beta=3, k_min=3, SR_min=0.04, range_x=None, range_y=None, range_z=None, show_3D=True)

    lidar_filtered_3d = o3d.geometry.PointCloud()
    lidar_filtered_3d.points = o3d.utility.Vector3dVector(lidar[filtered])
    lidar_filtered_3d.colors = o3d.utility.Vector3dVector(np.ones((len(lidar[filtered]), 3)) * 0.5)
    o3d.visualization.draw_geometries([lidar_filtered_3d])
