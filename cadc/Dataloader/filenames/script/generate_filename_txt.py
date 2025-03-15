import pandas as pd

cadc_stats = pd.read_csv('../../../cadc_dataset_route_stats.csv', header=0, usecols=[0, 1, 2, 18, 19, 20, 21])

rows = [0, 13, 32]

for row in rows:
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
            depth_dror_path.append(date + '/' + format(seq, '04') + "/labeled/image_0" + str(cam) +
                                   "/depth_dror/" + format(frame, '010') + ".png")

        with open('../inference_files.txt', 'a+') as f:
            for path in img_path:
                f.write('%s ' % path)
            for path in depth_path:
                f.write('%s ' % path)
            for path in depth_dror_path:
                f.write('%s ' % path)
            f.write('\n')
        #
        # if cadc_stats.iloc[row, 4] == 'Covered':
        #     with open('files_roadCover.txt', 'a+') as f:
        #         for path in img_path:
        #             f.write('%s ' % path)
        #         for path in depth_path:
        #             f.write('%s ' % path)
        #         for path in depth_dror_path:
        #             f.write('%s ' % path)
        #         f.write('\n')
        # elif cadc_stats.iloc[row, 4] == 'None':
        #     with open('files_roadNoCover.txt', 'a+') as f:
        #         for path in img_path:
        #             f.write('%s ' % path)
        #         for path in depth_path:
        #             f.write('%s ' % path)
        #         for path in depth_dror_path:
        #             f.write('%s ' % path)
        #         f.write('\n')
        #
        # if cadc_stats.iloc[row, 5] == 'Partial':
        #     with open('files_cam0Cover.txt', 'a+') as f:
        #         for path in img_path:
        #             f.write('%s ' % path)
        #         for path in depth_path:
        #             f.write('%s ' % path)
        #         for path in depth_dror_path:
        #             f.write('%s ' % path)
        #         f.write('\n')
        # elif cadc_stats.iloc[row, 5] == 'None':
        #     with open('files_cam0NoCover.txt', 'a+') as f:
        #         for path in img_path:
        #             f.write('%s ' % path)
        #         for path in depth_path:
        #             f.write('%s ' % path)
        #         for path in depth_dror_path:
        #             f.write('%s ' % path)
        #         f.write('\n')
        #
        # if cadc_stats.iloc[row, 6] == 'Light':
        #     with open('files_Light.txt', 'a+') as f:
        #         for path in img_path:
        #             f.write('%s ' % path)
        #         for path in depth_path:
        #             f.write('%s ' % path)
        #         for path in depth_dror_path:
        #             f.write('%s ' % path)
        #         f.write('\n')
        # elif cadc_stats.iloc[row, 6] == 'Medium':
        #     with open('files_Medium.txt', 'a+') as f:
        #         for path in img_path:
        #             f.write('%s ' % path)
        #         for path in depth_path:
        #             f.write('%s ' % path)
        #         for path in depth_dror_path:
        #             f.write('%s ' % path)
        #         f.write('\n')
        # elif cadc_stats.iloc[row, 6] == 'Heavy':
        #     with open('files_Heavy.txt', 'a+') as f:
        #         for path in img_path:
        #             f.write('%s ' % path)
        #         for path in depth_path:
        #             f.write('%s ' % path)
        #         for path in depth_dror_path:
        #             f.write('%s ' % path)
        #         f.write('\n')
        # elif cadc_stats.iloc[row, 6] == 'Extreme':
        #     with open('files_Extreme.txt', 'a+') as f:
        #         for path in img_path:
        #             f.write('%s ' % path)
        #         for path in depth_path:
        #             f.write('%s ' % path)
        #         for path in depth_dror_path:
        #             f.write('%s ' % path)
        #         f.write('\n')
