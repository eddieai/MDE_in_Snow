import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from time import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from cadc.dataset import DataGenerator
import net
import utils


def inference(model, dataloader, param):
    model.eval()

    # t = np.empty(0)
    for i, data in enumerate(dataloader):
        # t_start = time()

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

        # t_end = time()
        # print('%d of %d, inference time: %.3f' % (i+1, len(dataloader), t_end - t_start))
        # t = np.append(t, t_end - t_start)

        for j in range(len(pred_batch)):
            fig1 = plt.figure(1)
            ax1 = fig1.subplots(2, 1)
            ax1[0].clear()
            ax1[0].imshow(img_batch[j].cpu().permute(1, 2, 0))
            ax1[0].get_xaxis().set_visible(False)
            ax1[0].get_yaxis().set_visible(False)
            ax1[1].clear()
            ax1[1].imshow(pred_batch[j], cmap='jet')
            ax1[1].get_xaxis().set_visible(False)
            ax1[1].get_yaxis().set_visible(False)
            plt.savefig('inference_weighted_sigmoid_minent_1/' + format(i*len(pred_batch) + j, '03') + '.png')


model_dir = "experiments/refine_CADC_depth_aggregated_sord_weighted_sigmoid_minent_16x16_nomask_1_lr_1e-03_momentum_0.9_wd_0_epoch_30_mode_sord_pretrained_DeepLabV3+_Kitti_crop_513*513/best.pth.tar"

param = torch.load(model_dir).copy()
param['eval_n_crop'] = 10
param['batch_size'] = 3
param.pop('state_dict')
param.pop('optim_dict')
param.pop('sched_dict', None)
param.pop('restore_file', None)
param.pop('model_dir', None)

# data_gen_test = DataGenerator('/home/datasets/Kitti/', phase='test')
data_gen_test = DataGenerator('/home/datasets/CADC/cadcd/', '/home/datasets_mod/CADC/cadcd/', phase='inference', cam=0, depth_mode='dror')
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

# fig = plt.figure()
# plt.hist(t_inference, bins=20)
# plt.show()

# print('inference time average: %.3f\ninference time variance: %.3f' % (np.mean(t_inference), np.var(t_inference)))
