dir = '../'
stages = ['roadNoCover', 'roadCover']

all_train = []
all_val = []
all_test = []

for stage in stages:
    with open(dir + 'train_files_%s.txt' % stage, "r") as f:
        all_train += f.read().split('\n')
    with open(dir + 'val_files_%s.txt' % stage, "r") as f:
        all_val += f.read().split('\n')
    with open(dir + 'test_files_%s.txt' % stage, "r") as f:
        all_test += f.read().split('\n')

with open(dir + 'train_files.txt', 'w+') as f:
    f.write('\n'.join(all_train))
with open(dir + 'val_files.txt', 'w+') as f:
    f.write('\n'.join(all_val))
with open(dir + 'test_files.txt', 'w+') as f:
    f.write('\n'.join(all_test))