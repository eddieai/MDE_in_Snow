from tqdm import *
import numpy as np
import pandas as pd
import PIL.Image as Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
# plt.ion()
_, ax = plt.subplots(3, 3)

BASE = "/home/datasets/CADC/cadcd/"
BASE_mod = "/home/datasets_mod/CADC/cadcd/"
cadc_stats = pd.read_csv('cadc_dataset_route_stats.csv', header=0, usecols=[0, 1, 2, 18, 19, 20, 21])
start_row = 0   # 25
start_frame = 0
cam = 0

for row in trange(start_row, len(cadc_stats)):
    print('\n------ Date %s, Sequence %d ------' % (cadc_stats.iloc[row, 0], cadc_stats.iloc[row, 1]))
    date = cadc_stats.iloc[row, 0]
    seq = cadc_stats.iloc[row, 1]
    n_frame = cadc_stats.iloc[row, 2]

    for frame in trange(start_frame, n_frame):
        # visualization
        img_path = BASE + date + '/' + format(seq, '04') + "/labeled" + "/image_0" + str(cam) + \
                   "/data/" + format(frame, '010') + ".png"
        img = plt.imread(img_path)
        depth_raw_path = BASE_mod + date + '/' + format(seq, '04') + "/labeled" + "/image_0" + str(cam) + \
                   "/depth/" + format(frame, '010') + ".png"
        depth_raw = np.array(Image.open(depth_raw_path), dtype=int).astype(np.float32) / 256.
        depth_dror_path = BASE_mod + date + '/' + format(seq, '04') + "/labeled" + "/image_0" + str(cam) + \
                   "/depth_dror/" + format(frame, '010') + ".png"
        depth_dror = np.array(Image.open(depth_dror_path), dtype=int).astype(np.float32) / 256.
        depth_aggregated_3_path = BASE_mod + date + '/' + format(seq, '04') + "/labeled" + "/image_0" + str(cam) + \
                   "/depth_aggregated_3/" + format(frame, '010') + ".png"
        depth_aggregated_3 = np.array(Image.open(depth_aggregated_3_path), dtype=int).astype(np.float32) / 256.
        depth_aggregated_path = BASE_mod + date + '/' + format(seq, '04') + "/labeled" + "/image_0" + str(cam) + \
                   "/depth_aggregated/" + format(frame, '010') + ".png"
        depth_aggregated = np.array(Image.open(depth_aggregated_path), dtype=int).astype(np.float32) / 256.
        depth_HPR_3DBox_path = BASE_mod + date + '/' + format(seq, '04') + "/labeled" + "/image_0" + str(cam) + \
                   "/depth_HPR_3DBox/" + format(frame, '010') + ".png"
        depth_HPR_3DBox = np.array(Image.open(depth_HPR_3DBox_path), dtype=int).astype(np.float32) / 256.
        depth_HPR_ConvexHull_path = BASE_mod + date + '/' + format(seq, '04') + "/labeled" + "/image_0" + str(cam) + \
                   "/depth_HPR_ConvexHull/" + format(frame, '010') + ".png"
        depth_HPR_ConvexHull = np.array(Image.open(depth_HPR_ConvexHull_path), dtype=int).astype(np.float32) / 256.
        depth_HPR_ProjectedKNN_path = BASE_mod + date + '/' + format(seq, '04') + "/labeled" + "/image_0" + str(cam) + \
                   "/depth_HPR_ProjectedKNN/" + format(frame, '010') + ".png"
        depth_HPR_ProjectedKNN = np.array(Image.open(depth_HPR_ProjectedKNN_path), dtype=int).astype(np.float32) / 256.
        depth_HPR_ProjectedKNN_99_path = BASE_mod + date + '/' + format(seq, '04') + "/labeled" + "/image_0" + str(cam) + \
                                      "/depth_HPR_ProjectedKNN_0.99/" + format(frame, '010') + ".png"
        depth_HPR_ProjectedKNN_99 = np.array(Image.open(depth_HPR_ProjectedKNN_99_path), dtype=int).astype(np.float32) / 256.

        ax[0,0].clear()
        ax[0,0].imshow(img)
        ax[0,0].set_title('RGB image')
        ax[0, 0].get_xaxis().set_visible(False)
        ax[0,0].get_yaxis().set_visible(False)
        ax[0,1].clear()
        ax[0,1].imshow(img)
        ax[0,1].imshow(np.ma.masked_where(depth_raw==0, depth_raw.clip(0,80)), cmap='jet', vmin=0, vmax=80, alpha=0.5)
        ax[0,1].set_title('Raw Depth map')
        ax[0, 1].get_xaxis().set_visible(False)
        ax[0,1].get_yaxis().set_visible(False)
        ax[0,2].clear()
        ax[0,2].imshow(img)
        ax[0,2].imshow(np.ma.masked_where(depth_dror==0, depth_dror.clip(0,80)), cmap='jet', vmin=0, vmax=80, alpha=0.5)
        ax[0,2].set_title('DROR Depth map')
        ax[0, 2].get_xaxis().set_visible(False)
        ax[0,2].get_yaxis().set_visible(False)

        ax[1,0].clear()
        ax[1,0].imshow(img)
        ax[1,0].imshow(np.ma.masked_where(depth_aggregated_3==0, depth_aggregated_3.clip(0,80)), cmap='jet', vmin=0, vmax=80, alpha=0.5)
        ax[1,0].set_title('Aggregated 3 frames')
        ax[1, 0].get_xaxis().set_visible(False)
        ax[1,0].get_yaxis().set_visible(False)
        ax[1,1].clear()
        ax[1,1].imshow(img)
        ax[1,1].imshow(np.ma.masked_where(depth_aggregated==0, depth_aggregated.clip(0,80)), cmap='jet', vmin=0, vmax=80, alpha=0.5)
        ax[1,1].set_title('Aggregated 11 frames')
        ax[1,1].get_xaxis().set_visible(False)
        ax[1,1].get_yaxis().set_visible(False)
        ax[1,2].clear()
        ax[1,2].imshow(img)
        ax[1,2].imshow(np.ma.masked_where(depth_HPR_3DBox==0, depth_HPR_3DBox.clip(0,80)), cmap='jet', vmin=0, vmax=80, alpha=0.5)
        ax[1,2].set_title('HPR 3DBox')
        ax[1,2].get_xaxis().set_visible(False)
        ax[1,2].get_yaxis().set_visible(False)
        
        ax[2,0].clear()
        ax[2,0].imshow(img)
        ax[2,0].imshow(np.ma.masked_where(depth_HPR_ConvexHull==0, depth_HPR_ConvexHull.clip(0,80)), cmap='jet', vmin=0, vmax=80, alpha=0.5)
        ax[2,0].set_title('HPR ConvexHull')
        ax[2,0].get_xaxis().set_visible(False)
        ax[2,0].get_yaxis().set_visible(False)
        ax[2,1].clear()
        ax[2,1].imshow(img)
        ax[2,1].imshow(np.ma.masked_where(depth_HPR_ProjectedKNN==0, depth_HPR_ProjectedKNN.clip(0,80)), cmap='jet', vmin=0, vmax=80, alpha=0.5)
        ax[2,1].set_title('HPR ProjectedKNN mean')
        ax[2,1].get_xaxis().set_visible(False)
        ax[2,1].get_yaxis().set_visible(False)
        ax[2,2].clear()
        ax[2,2].imshow(img)
        ax[2,2].imshow(np.ma.masked_where(depth_HPR_ProjectedKNN_99==0, depth_HPR_ProjectedKNN_99.clip(0,80)), cmap='jet', vmin=0, vmax=80, alpha=0.5)
        ax[2,2].set_title('HPR ProjectedKNN 0.99')
        ax[2,2].get_xaxis().set_visible(False)
        ax[2,2].get_yaxis().set_visible(False)

        # plt.pause(0.1)
        # plt.show()
        plt.savefig('visualize_depth2/Date_%s_Seq_%2d_Frame_%2d.png' % (date, seq, frame), dpi=600)
