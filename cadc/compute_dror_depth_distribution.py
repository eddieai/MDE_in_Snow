from tqdm import *
import numpy as np
import pandas as pd
import h5py
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
import matplotlib.cm as cm
matplotlib.use('Agg')
cmap = cm.get_cmap('jet')

BASE = "/home/datasets/CADC/cadcd/"
BASE_mod = "/home/datasets_mod/CADC/cadcd/"
cadc_stats = pd.read_csv('cadc_dataset_route_stats.csv', header=0, usecols=[0, 1, 2, 18, 19, 20, 21])
start_row = 0
cam = 0

a = 1
b = 80
K = 120
bins = [np.exp(np.log(a) + np.log(b / a) * i / K) for i in range(K)]
bins_all = np.array([0] + bins + [80])
center = (bins_all[1:] + bins_all[:-1]) / 2

# dror_hist = np.zeros((0, K+1))
#
# for row in trange(start_row, len(cadc_stats)):
#     # print('\n------ Date %s, Sequence %d ------' % (cadc_stats.iloc[row, 0], cadc_stats.iloc[row, 1]))
#     date = cadc_stats.iloc[row, 0]
#     seq = cadc_stats.iloc[row, 1]
#     n_frame = cadc_stats.iloc[row, 2]
#
#     for frame in trange(0, n_frame):
#         # print('Loading lidar %d of %d (%.0f%%)' % (frame + 1, n_frame, (frame + 1) / n_frame * 100))
#         depth_dror_path = BASE_mod + date + '/' + format(seq, '04') + "/labeled" + "/image_0" + str(cam) + \
#                           "/depth_dror/" + format(frame, '010') + ".png"
#         depth_dror = np.array(Image.open(depth_dror_path), dtype=int).astype(np.float32) / 256.
#         depth_dror_non_zero = depth_dror[depth_dror>0]
#         dror_hist = np.vstack((dror_hist, np.histogram(depth_dror_non_zero, bins=bins_all)[0].reshape(1,K+1)))

# np.save('dror_hist.npy', dror_hist)

dror_hist = np.load('dror_hist.npy')
dror_hist = np.sum(dror_hist, axis=0)

fig, ax = plt.subplots()
ax.bar(bins_all[:-1], dror_hist, width=np.diff(bins_all), align='edge', color=cmap(np.arange(K+1).astype(float) / (K+1)))
ax.set_xscale('log')
ax.set_xlim(1,80)
ax.xaxis.set_minor_locator(ticker.FixedLocator([1,2,3,4,5,6,7,8,9] + list(range(10, 81, 10))))
ax.xaxis.set_major_locator(ticker.NullLocator())
ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
ax.set_xlabel('Meter')
ax.set_ylabel('Depth value count')
ax.set_title('Distribution of depth DROR distances of all frames (121 classes)')
plt.savefig('dror_hist.png')
