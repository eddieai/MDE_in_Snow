from tqdm import *
import os
import pandas as pd

BASE_mod = "/home/datasets_mod/CADC/cadcd/"
cadc_stats = pd.read_csv('../../../cadc_dataset_route_stats.csv', header=0, usecols=[0, 1, 2, 18, 19, 20, 21])
cam = 0

for row in trange(0, len(cadc_stats)):
    # print('\n------ Date %s, Sequence %d ------' % (cadc_stats.iloc[row, 0], cadc_stats.iloc[row, 1]))
    date = cadc_stats.iloc[row, 0]
    seq = cadc_stats.iloc[row, 1]

    depth_aggregate_path = BASE_mod + date + '/' + format(seq, '04') + "/labeled/lidar_points"

    for fn in os.listdir(depth_aggregate_path):
        if 'agg_culling' in fn:
            os.rename(os.path.join(depth_aggregate_path, fn),
                      os.path.join(depth_aggregate_path, fn.replace('agg_culling', 'HPR_3DBox')))
