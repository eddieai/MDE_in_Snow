import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from filter_lidar_dror import DROR_filter

#
# cadc_stats = pd.read_csv('cadc_dataset_route_stats.csv', header=0, usecols=[0, 1, 2, 18, 19, 20, 21])
#
# all_removed, light_removed, medium_removed, heavy_removed, extreme_removed = [np.empty(0) for _ in range(5)]
# all_removed_cube, light_removed_cube, medium_removed_cube, heavy_removed_cube, extreme_removed_cube = [np.empty(0) for _ in range(5)]
#
# for row in range(0, len(cadc_stats)):
#     print('\n------ Date %s, Sequence %d, Snowfall type %s ------' % (cadc_stats.iloc[row, 0], cadc_stats.iloc[row, 1], cadc_stats.iloc[row, 6]))
#     BASE = "/home/datasets/CADC/cadcd/" + cadc_stats.iloc[row, 0] + "/"
#     seq = cadc_stats.iloc[row, 1]
#     snowfall = cadc_stats.iloc[row, 6]
#
#     for frame in range(cadc_stats.iloc[row, 2]):
#         print('%d of %d frames' % (frame+1, cadc_stats.iloc[row, 2]))
#         lidar_path = BASE + format(seq, '04') + "/labeled/lidar_points/data/" + format(frame, '010') + ".bin"
#
#         # read lidar
#         lidar = np.fromfile(lidar_path, dtype=np.float32).reshape((-1, 4))
#         # lidar denoise by DROR
#         dror_filter = DROR_filter(lidar, alpha=0.2, beta=3, k_min=3, SR_min=0.04)
#         dror_filter_cube = DROR_filter(lidar, alpha=0.2, beta=3, k_min=3, SR_min=0.04, range_x=(-4,4), range_y=(-4,4), range_z=(-3,10))
#         #
#         removed = len(lidar) - len(dror_filter)
#         removed_cube = len(lidar) - len(dror_filter_cube)
#         print('Number of Lidar points removed by DROR: ', removed)
#         print('Number of Lidar points removed by DROR in cube defined in CADC: ', removed_cube)
#
#         all_removed = np.append(all_removed, removed)
#         all_removed_cube = np.append(all_removed_cube, removed_cube)
#         if snowfall == 'Light':
#             light_removed = np.append(light_removed, removed)
#             light_removed_cube = np.append(light_removed_cube, removed_cube)
#         if snowfall == 'Medium':
#             medium_removed = np.append(medium_removed, removed)
#             medium_removed_cube = np.append(medium_removed_cube, removed_cube)
#         if snowfall == 'Heavy':
#             heavy_removed = np.append(heavy_removed, removed)
#             heavy_removed_cube = np.append(heavy_removed_cube, removed_cube)
#         if snowfall == 'Extreme':
#             extreme_removed = np.append(extreme_removed, removed)
#             extreme_removed_cube = np.append(extreme_removed_cube, removed_cube)
#
# print('All mean: %.2f, Light mean: %.2f, Medium mean: %.2f, Heavy mean: %.2f, Extreme mean: %.2f' % (np.mean(all_removed), np.mean(light_removed), np.mean(medium_removed), np.mean(heavy_removed), np.mean(extreme_removed)))
# print('All mean in cube: %.2f, Light mean in cube: %.2f, Medium mean in cube: %.2f, Heavy mean in cube: %.2f, Extreme mean in cube: %.2f' % (np.mean(all_removed_cube), np.mean(light_removed_cube), np.mean(medium_removed_cube), np.mean(heavy_removed_cube), np.mean(extreme_removed_cube)))
# print('All variance: %.2f, Light variance: %.2f, Medium variance: %.2f, Heavy variance: %.2f, Extreme variance: %.2f' % (np.var(all_removed), np.var(light_removed), np.var(medium_removed), np.var(heavy_removed), np.var(extreme_removed)))
# print('All variance in cube: %.2f, Light variance in cube: %.2f, Medium variance in cube: %.2f, Heavy variance in cube: %.2f, Extreme variance in cube: %.2f' % (np.var(all_removed_cube), np.var(light_removed_cube), np.var(medium_removed_cube), np.var(heavy_removed_cube), np.var(extreme_removed_cube)))
#
# np.save('all_lidar_points_removed_by_dror', all_removed)
# np.save('light_lidar_points_removed_by_dror', light_removed)
# np.save('medium_lidar_points_removed_by_dror', medium_removed)
# np.save('heavy_lidar_points_removed_by_dror', heavy_removed)
# np.save('extreme_lidar_points_removed_by_dror', extreme_removed)
# np.save('all_lidar_points_removed_by_dror_cube', all_removed_cube)
# np.save('light_lidar_points_removed_by_dror_cube', light_removed_cube)
# np.save('medium_lidar_points_removed_by_dror_cube', medium_removed_cube)
# np.save('heavy_lidar_points_removed_by_dror_cube', heavy_removed_cube)
# np.save('extreme_lidar_points_removed_by_dror_cube', extreme_removed_cube)
# print('\nSave done')
#

all_removed = np.load('all_lidar_points_removed_by_dror.npy')
light_removed = np.load('light_lidar_points_removed_by_dror.npy')
medium_removed = np.load('medium_lidar_points_removed_by_dror.npy')
heavy_removed = np.load('heavy_lidar_points_removed_by_dror.npy')
extreme_removed = np.load('extreme_lidar_points_removed_by_dror.npy')
all_removed_cube = np.load('all_lidar_points_removed_by_dror_cube.npy')
light_removed_cube = np.load('light_lidar_points_removed_by_dror_cube.npy')
medium_removed_cube = np.load('medium_lidar_points_removed_by_dror_cube.npy')
heavy_removed_cube = np.load('heavy_lidar_points_removed_by_dror_cube.npy')
extreme_removed_cube = np.load('extreme_lidar_points_removed_by_dror_cube.npy')

print('All mean: %.2f, Light mean: %.2f, Medium mean: %.2f, Heavy mean: %.2f, Extreme mean: %.2f' % (np.mean(all_removed), np.mean(light_removed), np.mean(medium_removed), np.mean(heavy_removed), np.mean(extreme_removed)))
print('All mean in cube: %.2f, Light mean in cube: %.2f, Medium mean in cube: %.2f, Heavy mean in cube: %.2f, Extreme mean in cube: %.2f' % (np.mean(all_removed_cube), np.mean(light_removed_cube), np.mean(medium_removed_cube), np.mean(heavy_removed_cube), np.mean(extreme_removed_cube)))
print('All variance: %.2f, Light variance: %.2f, Medium variance: %.2f, Heavy variance: %.2f, Extreme variance: %.2f' % (np.var(all_removed), np.var(light_removed), np.var(medium_removed), np.var(heavy_removed), np.var(extreme_removed)))
print('All variance in cube: %.2f, Light variance in cube: %.2f, Medium variance in cube: %.2f, Heavy variance in cube: %.2f, Extreme variance in cube: %.2f' % (np.var(all_removed_cube), np.var(light_removed_cube), np.var(medium_removed_cube), np.var(heavy_removed_cube), np.var(extreme_removed_cube)))

_, ax = plt.subplots(1, 5, sharex=True, sharey=True)
ax[0].hist(all_removed, bins=range(0, 20000+200, 200))
ax[0].set_title('All lidar removed')
ax[0].set_xlim(0, 20000)
ax[1].hist(light_removed, bins=range(0, 20000+200, 200))
ax[1].set_title('Light snow lidar removed')
ax[1].set_xlim(0, 20000)
ax[2].hist(medium_removed, bins=range(0, 20000+200, 200))
ax[2].set_title('Medium snow lidar removed')
ax[2].set_xlim(0, 20000)
ax[3].hist(heavy_removed, bins=range(0, 20000+200, 200))
ax[3].set_title('Heavy snow lidar removed')
ax[3].set_xlim(0, 20000)
ax[4].hist(extreme_removed, bins=range(0, 20000+200, 200))
ax[4].set_title('Extreme snow lidar removed')
ax[4].set_xlim(0, 20000)

_, ax = plt.subplots(1, 5, sharex=True, sharey=True)
ax[0].hist(all_removed_cube, bins=range(0, 2000+20, 20))
ax[0].set_title('All lidar removed in cube')
ax[0].set_xlim(0, 2000)
ax[1].hist(light_removed_cube, bins=range(0, 2000+20, 20))
ax[1].set_title('Light snow lidar removed in cube')
ax[1].set_xlim(0, 2000)
ax[2].hist(medium_removed_cube, bins=range(0, 2000+20, 20))
ax[2].set_title('Medium snow lidar removed in cube')
ax[2].set_xlim(0, 2000)
ax[3].hist(heavy_removed_cube, bins=range(0, 2000+20, 20))
ax[3].set_title('Heavy snow lidar removed in cube')
ax[3].set_xlim(0, 2000)
ax[4].hist(extreme_removed_cube, bins=range(0, 2000+20, 20))
ax[4].set_title('Extreme snow lidar removed in cube')
ax[4].set_xlim(0, 2000)
plt.show()
