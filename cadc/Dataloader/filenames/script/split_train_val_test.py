import random

dir = '../'
txt_name = 'all_files.txt'

with open(dir + txt_name, "r") as f:
    data = f.read().split('\n')[:-1]

random.shuffle(data)

# 80-10-10
train_data = data[:int(len(data)*0.8)]
val_data = data[int(len(data)*0.8):int(len(data)*0.9)]
test_data = data[int(len(data)*0.9):]

with open(dir + 'train' + txt_name[3:], 'w+') as f:
    f.write('\n'.join(train_data))
with open(dir + 'val' + txt_name[3:], 'w+') as f:
    f.write('\n'.join(val_data))
with open(dir + 'test' + txt_name[3:], 'w+') as f:
    f.write('\n'.join(test_data))
