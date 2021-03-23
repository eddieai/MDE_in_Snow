import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import torch
from cadc.dataset import DataGenerator
import net
import utils


def inference(model, dataloader, param):
    model.eval()

    BASE_mod = "/home/datasets_mod/CADC/cadcd/"
    f = open('cadc/Dataloader/filenames/all_seq_files.txt', 'r')
    all_seq_files = f.readlines()

    for i, data in enumerate(dataloader):
        img_batch = data['img']
        depth_batch = data['depth']
        # move to GPU if available
        img_batch_cropped = net.equally_spaced_crop(img_batch, param['eval_n_crop'])
        img_batch_cropped = img_batch_cropped.cuda()

        with torch.no_grad():
            # compute model output
            output_batch_cropped = model(img_batch_cropped)

        output_batch_cropped, depth_batch = output_batch_cropped.cpu().numpy(), depth_batch.cpu().numpy()
        pred_batch_cropped = net.depth_inference(output_batch_cropped, param['mode'])
        pred_batch = net.pred_overlap(pred_batch_cropped, depth_batch.shape, param['eval_n_crop'])

        for j in range(len(pred_batch)):
            file_idx = i * len(pred_batch) + j
            date = all_seq_files[file_idx][:10]
            seq = all_seq_files[file_idx][11:15]
            frame = all_seq_files[file_idx][38:48]

            print('Depth inference of Date %s Sequence %d Frame %d' % (date, int(seq), int(frame)))

            depth_inference_path = BASE_mod + date + '/' + seq + "/labeled/image_00/depth_inference_pretrained_Kitti/" + frame + ".png"
            if not (os.path.exists(depth_inference_path[:-14])):
                os.makedirs(depth_inference_path[:-14])
            depth_inference_PIL = Image.fromarray(np.clip(pred_batch[j] * 256., 0, 65535)).convert('I')
            depth_inference_PIL.save(depth_inference_path, mode='I;16')


model_dir = "experiments/train_lr_1e-03_momentum_0.9_wd_0_epoch_30_mode_sord_pretrained_DeepLabV3+_PascalVOC_crop_375*513/best.pth.tar"

param = torch.load(model_dir).copy()
param['eval_n_crop'] = 4
param['batch_size'] = 3
param.pop('state_dict')
param.pop('optim_dict')
param.pop('sched_dict', None)
param.pop('restore_file', None)
param.pop('model_dir', None)

# data_gen_test = DataGenerator('/home/datasets/Kitti/', phase='test')
data_gen_test = DataGenerator('/home/datasets/CADC/cadcd/', '/home/datasets_mod/CADC/cadcd/', phase='all_seq', cam=0, depth_mode='dror')
print('val or test data size:', len(data_gen_test.dataset))
dataloader_test = data_gen_test.create_data(batch_size=param['batch_size'])
data = next(iter(dataloader_test))
print('img shape:', data['img'].shape)
print('depth shape:', data['depth'].shape)

model = net.get_model(param['mode'], pretrained=False)
model.cuda()

utils.load_checkpoint(model_dir, model)
print('Load model parameters done')

inference(model, dataloader_test, param)
