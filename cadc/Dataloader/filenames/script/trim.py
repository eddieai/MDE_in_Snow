import os

root_dir = '../'
files = filter(lambda x: x[-4:] == '.txt', os.listdir(root_dir))

for file in sorted(files):

    with open(root_dir + file, 'r') as f_old:
        data = f_old.read()

    data = data.replace('  ', ' ')
    data = data.replace('\n \n', '\n')

    with open(root_dir + file[:-4] + '_new.txt', 'w+') as f_new:
        f_new.write(data)
