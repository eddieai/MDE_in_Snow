import pandas as pd
from random import sample

cadc_stats = pd.read_csv('../../../cadc_dataset_route_stats.csv', header=0, usecols=[0, 1, 2, 18, 19, 20, 21])

train_rows, val_rows, test_rows = [], [], []
dates = ['2018_03_06', '2018_03_07', '2019_02_27']
for date in dates:
    rows = cadc_stats[cadc_stats.Date == date].index
    n_row = len(rows)
    random_rows = sample(list(rows), n_row)
    train_rows += random_rows[:int(n_row*0.8)]
    test_rows += random_rows[int(n_row*0.8):int(n_row*0.9)]
    val_rows += random_rows[int(n_row*0.9):]

for row in train_rows:
    print('\n------ Date %s, Sequence %d ------' % (cadc_stats.iloc[row, 0], cadc_stats.iloc[row, 1]))
    date = cadc_stats.iloc[row, 0]
    seq = cadc_stats.iloc[row, 1]

    for frame in range(cadc_stats.iloc[row, 2]):
        img_path = []
        depth_path = []
        depth_dror_path = []
        for cam in range(8):
            img_path.append(date + '/' + format(seq, '04') + "/labeled/image_0" + str(cam) +
                            "/data/" + format(frame, '010') + ".png")
            depth_path.append(date + '/' + format(seq, '04') + "/labeled/image_0" + str(cam) +
                              "/depth/" + format(frame, '010') + ".png")
            # depth_dror_path.append(date + '/' + format(seq, '04') + "/labeled/image_0" + str(cam) +
            #                        "/depth_dror/" + format(frame, '010') + ".png")

        with open('../train_seq_files.txt', 'a+') as f:
            for path in img_path:
                f.write('%s ' % path)
            for path in depth_path:
                f.write('%s ' % path)
            # for path in depth_dror_path:
            #     f.write('%s ' % path)
            f.write('\n')

for row in val_rows:
    print('\n------ Date %s, Sequence %d ------' % (cadc_stats.iloc[row, 0], cadc_stats.iloc[row, 1]))
    date = cadc_stats.iloc[row, 0]
    seq = cadc_stats.iloc[row, 1]

    for frame in range(cadc_stats.iloc[row, 2]):
        img_path = []
        depth_path = []
        depth_dror_path = []
        for cam in range(8):
            img_path.append(date + '/' + format(seq, '04') + "/labeled/image_0" + str(cam) +
                            "/data/" + format(frame, '010') + ".png")
            depth_path.append(date + '/' + format(seq, '04') + "/labeled/image_0" + str(cam) +
                              "/depth/" + format(frame, '010') + ".png")
            # depth_dror_path.append(date + '/' + format(seq, '04') + "/labeled/image_0" + str(cam) +
            #                        "/depth_dror/" + format(frame, '010') + ".png")

        with open('../val_seq_files.txt', 'a+') as f:
            for path in img_path:
                f.write('%s ' % path)
            for path in depth_path:
                f.write('%s ' % path)
            # for path in depth_dror_path:
            #     f.write('%s ' % path)
            f.write('\n')

for row in test_rows:
    print('\n------ Date %s, Sequence %d ------' % (cadc_stats.iloc[row, 0], cadc_stats.iloc[row, 1]))
    date = cadc_stats.iloc[row, 0]
    seq = cadc_stats.iloc[row, 1]

    for frame in range(cadc_stats.iloc[row, 2]):
        img_path = []
        depth_path = []
        depth_dror_path = []
        for cam in range(8):
            img_path.append(date + '/' + format(seq, '04') + "/labeled/image_0" + str(cam) +
                            "/data/" + format(frame, '010') + ".png")
            depth_path.append(date + '/' + format(seq, '04') + "/labeled/image_0" + str(cam) +
                              "/depth/" + format(frame, '010') + ".png")
            # depth_dror_path.append(date + '/' + format(seq, '04') + "/labeled/image_0" + str(cam) +
            #                        "/depth_dror/" + format(frame, '010') + ".png")

        with open('../test_seq_files.txt', 'a+') as f:
            for path in img_path:
                f.write('%s ' % path)
            for path in depth_path:
                f.write('%s ' % path)
            # for path in depth_dror_path:
            #     f.write('%s ' % path)
            f.write('\n')