import numpy as np
import pandas as pd
import PIL.Image as Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
plt.ion()

from skimage.filters.rank import entropy
from scipy.stats import entropy as scipyentropy


BASE = "/home/datasets/CADC/cadcd/"
BASE_mod = "/home/datasets_mod/CADC/cadcd/"
cadc_stats = pd.read_csv('cadc_dataset_route_stats.csv', header=0, usecols=[0, 1, 2, 18, 19, 20, 21])
start_row = 25  # 25
cam = 0

for row in range(start_row, len(cadc_stats)):
    print('\n------ Date %s, Sequence %d ------' % (cadc_stats.iloc[row, 0], cadc_stats.iloc[row, 1]))
    date = cadc_stats.iloc[row, 0]
    seq = cadc_stats.iloc[row, 1]
    n_frame = cadc_stats.iloc[row, 2]
    start_frame = 10

    for frame in range(start_frame, n_frame):
        # visualization
        img_path = BASE + date + '/' + format(seq, '04') + "/labeled" + "/image_0" + str(cam) + \
                   "/data/" + format(frame, '010') + ".png"
        img = plt.imread(img_path)
        img_grayscale = np.array(Image.open(img_path).convert("L"))
        img_entropy = entropy(img_grayscale.astype(np.uint8), np.ones((16,16)).astype(np.uint8))

        depth_aggregate_path = BASE_mod + date + '/' + format(seq, '04') + "/labeled" + "/image_0" + str(cam) + \
                   "/depth_aggregated/" + format(frame, '010') + ".png"
        depth_aggregate = np.array(Image.open(depth_aggregate_path), dtype=int).astype(np.float32) / 256.
        depth_aggregate_entropy = entropy(depth_aggregate.clip(0,80).astype(np.uint8), selem=np.ones((16,16)).astype(np.uint8), mask=(depth_aggregate>0))

        # # Digitize depth into 10 bins of log space before compute entropy
        # depth_aggregate_log_digitize = np.digitize(depth_aggregate.clip(0, 80), bins=[np.exp(np.log(1) + np.log(80/1) * i / 9) for i in range(9)])
        # depth_aggregate_digitize = np.digitize(depth_aggregate.clip(0, 80), bins=np.arange(8,80,8))
        # depth_aggregate_entropy_2 = entropy(depth_aggregate_log_digitize.astype(np.uint8), selem=np.ones((3,3)).astype(np.uint8), mask=(depth_aggregate>0))

        # # Manually compute local entropy map
        # m,n = 16,16
        # depth_aggregate_entropy_1 = np.zeros((1024,1280))
        # for i in range(depth_aggregate.shape[0]):
        #     for j in range(depth_aggregate.shape[1]):
        #         patch = np.clip(depth_aggregate[i:min(i+m,1024), j:min(j+n,1280)], 0, 80)
        #         if np.all(patch==0):
        #             depth_aggregate_entropy_1[i, j] = 0
        #         else:
        #             patch_nonzero = patch[patch>0]
        #             hist_patch = np.histogram(patch_nonzero, bins=256, range=(0,256))[0] / len(patch_nonzero)
        #             depth_aggregate_entropy_1[i,j] = scipyentropy(hist_patch, base=2)

        # plot distribution of entropy values
        print(np.mean(depth_aggregate_entropy[depth_aggregate>0]), np.median(depth_aggregate_entropy[depth_aggregate>0]))
        fig0 = plt.figure(0)
        plt.cla()
        plt.hist(depth_aggregate_entropy[depth_aggregate>0].flatten(), bins=100)

        fig1 = plt.figure(1)
        ax1 = fig1.subplots(1,4)
        ax1[0].clear()
        ax1[0].imshow(img)
        # ax1[0].add_patch(Rectangle((0, 0), 16, 16, fill=True, linewidth=0, facecolor='w'))
        ax1[0].set_title('RGB image')
        ax1[0].get_xaxis().set_visible(False)
        ax1[0].get_yaxis().set_visible(False)

        ax1[1].clear()
        ax1[1].imshow(np.clip(depth_aggregate, 0, 80), cmap='jet')
        # ax1[1].add_patch(Rectangle((0, 0), 16, 16, fill=True, linewidth=0, facecolor='w'))
        ax1[1].set_title('Aggregated Depth map')
        ax1[1].get_xaxis().set_visible(False)
        ax1[1].get_yaxis().set_visible(False)

        ax1[2].clear()
        ax1[2].imshow(depth_aggregate_entropy, cmap='gray')
        # ax1[2].add_patch(Rectangle((0, 0), 16, 16, fill=True, linewidth=0, facecolor='w'))
        ax1[2].set_title('Local Entropy of depth map')
        ax1[2].get_xaxis().set_visible(False)
        ax1[2].get_yaxis().set_visible(False)

        ax1[3].clear()
        ax1[3].imshow(np.where(depth_aggregate>0, np.clip(1 - depth_aggregate_entropy/6, 0, np.inf), 0), cmap='viridis')
        ax1[3].set_title('Confidence map')
        ax1[3].get_xaxis().set_visible(False)
        ax1[3].get_yaxis().set_visible(False)

        #
        #
        # fig2 = plt.figure(2)
        # ax2 = fig2.subplots(2,2)
        # ax2[0,0].clear()
        # ax2[0,0].imshow(np.where(depth_aggregate>0, 1 - 1/(1+np.exp(-depth_aggregate_entropy)), 0), cmap='viridis')
        # ax2[0,0].set_title('1 - sigmoid(entropy)')
        #
        # ax2[0,1].clear()
        # ax2[0,1].imshow(np.where(depth_aggregate>0, 1 - depth_aggregate_entropy/depth_aggregate_entropy.max(), 0), cmap='viridis')
        # ax2[0,1].set_title('1 - entropy/entropy.max()')
        #
        # ax2[1,0].clear()
        # ax2[1,0].imshow(np.where(depth_aggregate>0, np.exp(-depth_aggregate_entropy), 0), cmap='viridis')
        # ax2[1,0].set_title('exp(-entropy)')
        #
        # ax2[1,1].clear()
        # ax2[1,1].imshow(np.where(depth_aggregate>0, 1/depth_aggregate_entropy, 0), cmap='viridis')
        # ax2[1,1].set_title('1/entropy')

        plt.pause(0.1)
        plt.show()
