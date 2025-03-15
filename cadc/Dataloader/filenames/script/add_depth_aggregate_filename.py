import os

root_dir = './'
files = filter(lambda x: x[-4:] == '.txt', os.listdir(root_dir))

for file in sorted(files):

    with open(root_dir + file, 'r') as f_old:
        data = f_old.read().split('\n')

    for i, row in enumerate(data):
        path = row.split(' ')[:-1]
        path_depth_dror = path[-8:]
        path_depth_aggregate = map(lambda path: path.replace('dror', 'aggregated'), path_depth_dror)
        path += path_depth_aggregate

        with open(root_dir + file[:-4] + '_new.txt', 'a+') as f_new:
            for item in path:
                f_new.write('%s ' % item)
            f_new.write('\n')
