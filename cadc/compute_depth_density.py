import numpy as np
import pandas as pd
from PIL import Image

filepath = "Dataloader/filenames/all_files.txt"
Calculate_Raw = False
Calculate_DROR = False
Calculate_Aggregated = True
VISUALIZE = True

with open(filepath, "r") as f:
    data_list = f.read().split('\n')

if Calculate_Raw:
    density_raw_list = np.empty(0)
if Calculate_DROR:
    density_dror_list = np.empty(0)
if Calculate_Aggregated:
    density_aggregated_list = np.empty(0)
    cadc_stats_path = './cadc_dataset_route_stats.csv'
    cadc_stats = pd.read_csv(cadc_stats_path, header=0, usecols=[0, 1, 2, 18, 19, 20, 21])

if VISUALIZE:
    import matplotlib.pyplot as plt
    import matplotlib

cam = 0
for i, data in enumerate(data_list):
    if data != '':
        date = data[:10]
        seq = int(data[11:15])
        frame = int(data[-7:-5])
        print('%d of %d (%.0f%%), Date %s - Seq %d - Frame %d' %
              (i+1, len(data_list), (i + 1) / len(data_list) * 100, date, seq, frame))

        file = data.split(' ')
        if Calculate_Raw:
            depth_raw = np.array(Image.open('/home/datasets_mod/CADC/cadcd/' + file[cam + 8]), dtype=int)
            density_raw = np.count_nonzero(depth_raw[200:200+513, :]) / (513*1280)
            density_raw_list = np.append(density_raw_list, density_raw)
            print('\tDensity Raw = %.3f%%' % (density_raw * 100))
        if Calculate_DROR:
            depth_dror = np.array(Image.open('/home/datasets_mod/CADC/cadcd/' + file[cam + 16]), dtype=int)
            density_dror = np.count_nonzero(depth_dror[200:200+513, :]) / (513*1280)
            density_dror_list = np.append(density_dror_list, density_dror)
            print('\tDensity DROR = %.3f%%' % (density_dror * 100))
        if Calculate_Aggregated:
            frame_count = cadc_stats[(cadc_stats.Date == date) &
                                     (cadc_stats.Number == seq)].iloc[:, 2].values
            if not (5 <= frame < frame_count - 5):
                continue
            depth_aggregated = np.array(Image.open('/home/datasets_mod/CADC/cadcd/' + file[cam + 24]), dtype=int)
            density_aggregated = np.count_nonzero(depth_aggregated[200:200 + 513, :]) / (513 * 1280)
            density_aggregated_list = np.append(density_aggregated_list, density_aggregated)
            print('\tDensity Aggregated = %.3f%%' % (density_aggregated * 100))

print()
if Calculate_Raw:
    np.save('cam0_depth_density_raw.npy', density_raw_list)
    density_raw_mean = np.mean(density_raw_list)
    density_raw_median = np.median(density_raw_list)
    print('cam0 Mean of Density Raw = %.3f%%, Median of Density Raw = %.3f%%' %
          (density_raw_mean * 100, density_raw_median * 100))
if Calculate_DROR:
    np.save('cam0_depth_density_dror.npy', density_dror_list)
    density_dror_mean = np.mean(density_dror_list)
    density_dror_median = np.median(density_dror_list)
    print('cam0 Mean of Density DROR = %.3f%%, Median of Density DROR = %.3f%%' %
          (density_dror_mean * 100, density_dror_median * 100))
if Calculate_Aggregated:
    np.save('cam0_depth_density_aggregated.npy', density_aggregated_list)
    density_aggregated_mean = np.mean(density_aggregated_list)
    density_aggregated_median = np.median(density_aggregated_list)
    print('cam0 Mean of Density Aggregated = %.3f%%, Median of Density Aggregated = %.3f%%' %
          (density_aggregated_mean * 100, density_aggregated_median * 100))


if VISUALIZE:
    density_raw_list = np.load('cam0_depth_density_raw.npy')
    density_dror_list = np.load('cam0_depth_density_dror.npy')
    density_aggregated_list = np.load('cam0_depth_density_aggregated.npy')
    density_aggregated_NoSmart_list = np.load('cam0_depth_density_aggregated_cam0_depth_density_aggregated_noculling_old.npy')

    font = {'family' : 'sans',
            'size'   : 22}
    matplotlib.rc('font', **font)

    _, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    ax[0].hist(density_raw_list, bins=100)
    ax[0].set_xlabel('Raw')
    ax[1].hist(density_dror_list, bins=100)
    ax[1].set_xlabel('After DROR')

    _, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    ax[0].hist(density_aggregated_NoSmart_list, bins=100)
    ax[0].set_xlabel('Aggregate No filter (11 frames)')
    ax[1].hist(density_aggregated_list, bins=100)
    ax[1].set_xlabel('Smart Aggregate (11 frames)')

    plt.show()

    density_raw_mean = np.mean(density_raw_list)
    density_raw_median = np.median(density_raw_list)
    print('cam0 Mean of Density Raw = %.3f%%, Median of Density Raw = %.3f%%' %
          (density_raw_mean * 100, density_raw_median * 100))
    density_dror_mean = np.mean(density_dror_list)
    density_dror_median = np.median(density_dror_list)
    print('cam0 Mean of Density DROR = %.3f%%, Median of Density DROR = %.3f%%' %
          (density_dror_mean * 100, density_dror_median * 100))
    density_aggregated_NoSmart_mean = np.mean(density_aggregated_NoSmart_list)
    density_aggregated_NoSmart_median = np.median(density_aggregated_NoSmart_list)
    print('cam0 Mean of Density Aggregated No Smart = %.3f%%, Median of Density Aggregated No Smart = %.3f%%' %
          (density_aggregated_NoSmart_mean * 100, density_aggregated_NoSmart_median * 100))
    density_aggregated_mean = np.mean(density_aggregated_list)
    density_aggregated_median = np.median(density_aggregated_list)
    print('cam0 Mean of Density Smart Aggregated = %.3f%%, Median of Density Smart Aggregated = %.3f%%' %
          (density_aggregated_mean * 100, density_aggregated_median * 100))
