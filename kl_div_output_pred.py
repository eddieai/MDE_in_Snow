import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
from PIL import Image
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib.cm as cm
import torch
from cadc.dataset import DataGenerator
import net
import utils

# matplotlib.use('TkAgg')
# cmap = cm.get_cmap('jet')
# plt.ion()
#
# fig2 = plt.figure(2)
# axes = fig2.subplots(1, 3, sharex=True, sharey=True)


def pseudo_label(model, dataloader, param):
    model.eval()

    BASE_mod = "/home/datasets_mod/CADC/cadcd/"
    f = open('cadc/Dataloader/filenames/all_seq_files.txt', 'r')
    all_seq_files = f.readlines()

    # t = np.empty(0)
    for i, data in enumerate(dataloader):
        # t_start = time()

        img_batch = data['img']
        depth_batch = data['depth']
        img_batch_cropped = net.equally_spaced_crop(img_batch, param['eval_n_crop'])
        img_batch_cropped = img_batch_cropped.cuda()

        # depth_batch_cropped = net.equally_spaced_crop(depth_batch, param['eval_n_crop'])
        # mask = (depth_batch_cropped != 0)
        # gt = depth_batch_cropped.clamp(0, 80)[:, :, :, None]
        # # phi = (torch.log(gt) - torch.log(torch.tensor(center).float().cuda()).view(1,1,1,-1))**2
        center = torch.tensor(net.center).float()
        # phi_gt = (gt - center.view(1, 1, 1, -1)) ** 2
        # gt_sord = F.softmax(-phi_gt, dim=3)

        with torch.no_grad():
            # compute model output
            output_batch_cropped = model(img_batch_cropped)

        pred_batch_cropped = net.depth_inference(output_batch_cropped.cpu().numpy(), param['mode'])

        # pred_cropped = torch.sum(F.softmax(output_batch_cropped.cpu(), dim=1) * center.view(1, -1, 1, 1), dim=1)
        phi_pred = (torch.tensor(pred_batch_cropped).float().clamp(0, net.b)[:, :, :, None] - center.view(1, 1, 1, -1)) ** 2
        pred_cropped_sord = F.softmax(-phi_pred, dim=3)

        # p = F.softmax(output_batch_cropped.cpu(), dim=1).permute(0, 2, 3, 1)
        log_p = F.log_softmax(output_batch_cropped.cpu(), dim=1).permute(0, 2, 3, 1)
        kl_div_cropped = F.kl_div(log_p, pred_cropped_sord, reduction='none').sum(dim=3).numpy()

        # Plot Ground-truth SORD of a pixel and Output of the same pixel
        # axes[0].clear()
        # axes[0].bar(net.center, gt_sord[mask].numpy()[0], width=np.diff(np.append(net.center, [80])) / 2,
        #             align='edge', color=cmap(np.arange(net.K).astype(float) / net.K))
        # axes[0].set_title('A pixel\'s ground-truth SORD')
        # axes[0].set_facecolor('black')
        # axes[0].set_xscale('log')
        # axes[0].set_xlim(0.5, 80)
        # axes[0].set_ylim(0, 1)
        # # axes[0].xaxis.set_minor_locator(ticker.FixedLocator([1] + list(range(10, 81, 10))))
        # # axes[0].xaxis.set_major_locator(ticker.NullLocator())
        # # axes[0].xaxis.set_minor_formatter(ticker.ScalarFormatter())
        #
        # axes[1].clear()
        # axes[1].bar(net.center, p[mask].numpy()[0], width=np.diff(np.append(net.center, [80])) / 2,
        #             align='edge', color=cmap(np.arange(net.K).astype(float) / net.K))
        # axes[1].set_title('The pixel\'s network output')
        # axes[1].set_facecolor('black')
        #
        # axes[2].clear()
        # axes[2].bar(net.center, pred_cropped_sord[mask].numpy()[0], width=np.diff(np.append(net.center, [80])) / 2,
        #             align='edge', color=cmap(np.arange(net.K).astype(float) / net.K))
        # axes[2].set_title('The pixel\'s depth prediction SORD')
        # axes[2].set_facecolor('black')
        # axes[2].set_xlabel('KL-divergence of network ouput / depth prediction SORD = %.3f' % kl_div_cropped[mask][0])
        #
        # # plt.savefig('sord_pseudo_visualize/' + format(i*len(kl_div_batch) + j, '03') + '.png', dpi=300)
        # plt.pause(0.1)
        # plt.show()

        N, H, W = depth_batch.shape
        n_crop = param['eval_n_crop']

        kl_div_batch = np.zeros((N, H, W))
        overlap = np.zeros((H, W))
        equal_space = round((W - 513) / (n_crop - 1))

        for n in range(n_crop - 1):
            left = n * equal_space
            overlap[:, left:left + 513] += 1
            kl_div_batch[..., left:left + 513] += kl_div_cropped[n * N:(n + 1) * N, ...]
        overlap[..., -513:] += 1
        kl_div_batch[..., -513:] += kl_div_cropped[-N:, ...]

        kl_div_batch /= overlap

        # t_end = time()
        # print('%d of %d, inference time: %.3f' % (i+1, len(dataloader), t_end - t_start))
        # t = np.append(t, t_end - t_start)

        # for j in range(len(kl_div_batch)):
        #     fig1 = plt.figure(1)
        #     ax1 = fig1.subplots(4, 2)
        #     ax1[0,0].clear()
        #     ax1[0,0].imshow(img_batch[j].cpu().permute(1, 2, 0))
        #     ax1[0,0].get_xaxis().set_visible(False)
        #     ax1[0,0].get_yaxis().set_visible(False)
        #     ax1[0,0].set_title('RGB image')
        #     ax1[0,1].clear()
        #     ax1[0,1].imshow(depth_batch[j].cpu().clamp(0, net.b), vmin=0, vmax=net.b, cmap='jet')
        #     ax1[0,1].get_xaxis().set_visible(False)
        #     ax1[0,1].get_yaxis().set_visible(False)
        #     ax1[0,1].set_title('Depth map after ProjectedKNN')
        #     ax1[1,0].clear()
        #     ax1[1,0].imshow(pred_batch[j].clip(0, net.b), vmin=0, vmax=net.b, cmap='jet')
        #     ax1[1,0].get_xaxis().set_visible(False)
        #     ax1[1,0].get_yaxis().set_visible(False)
        #     ax1[1,0].set_title('Depth prediction')
        #     ax1[1,1].clear()
        #     ax1[1,1].imshow(kl_div_batch[j].clip(0, 1), vmin=0, vmax=1)
        #     ax1[1,1].get_xaxis().set_visible(False)
        #     ax1[1,1].get_yaxis().set_visible(False)
        #     ax1[1,1].set_title('Pixel-wise KL Divergence clip to (0, 1)')
        #     ax1[2,0].clear()
        #     ax1[2,0].imshow((kl_div_batch[j] < 1), cmap='gray')
        #     ax1[2,0].get_xaxis().set_visible(False)
        #     ax1[2,0].get_yaxis().set_visible(False)
        #     ax1[2,0].set_title('Pixel-wise KL Divergence < 1')
        #     ax1[2,1].clear()
        #     ax1[2,1].imshow((kl_div_batch[j] < 0.8), cmap='gray')
        #     ax1[2,1].get_xaxis().set_visible(False)
        #     ax1[2,1].get_yaxis().set_visible(False)
        #     ax1[2,1].set_title('Pixel-wise KL Divergence < 0.8')
        #     ax1[3,0].clear()
        #     ax1[3,0].imshow((kl_div_batch[j] < 0.5), cmap='gray')
        #     ax1[3,0].get_xaxis().set_visible(False)
        #     ax1[3,0].get_yaxis().set_visible(False)
        #     ax1[3,0].set_title('Pixel-wise KL Divergence < 0.5')
        #     ax1[3,1].clear()
        #     ax1[3,1].imshow((kl_div_batch[j] < 0.1), cmap='gray')
        #     ax1[3,1].get_xaxis().set_visible(False)
        #     ax1[3,1].get_yaxis().set_visible(False)
        #     ax1[3,1].set_title('Pixel-wise KL Divergence < 0.1')
        #
        #     plt.savefig('pseudo_label_KL_visualize/' + format(i*len(kl_div_batch) + j, '03') + '.png', dpi=300)

        for j in range(len(kl_div_batch)):
            file_idx = i * len(kl_div_batch) + j
            date = all_seq_files[file_idx][:10]
            seq = all_seq_files[file_idx][11:15]
            frame = all_seq_files[file_idx][38:48]

            print('KL-div (output / prediction) of Date %s Sequence %d Frame %d' % (date, int(seq), int(frame)))

            kl_div_path = BASE_mod + date + '/' + seq + "/labeled/image_00/kl_div_output_pred_pretrained_Kitti/" + frame + ".png"
            if not (os.path.exists(kl_div_path[:-14])):
                os.makedirs(kl_div_path[:-14])
            kl_div_PIL = Image.fromarray(np.clip(kl_div_batch[j] * 256., 0, 255)).convert('L')
            kl_div_PIL.save(kl_div_path, mode='L')


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
data_gen_test = DataGenerator('/home/datasets/CADC/cadcd/', '/home/datasets_mod/CADC/cadcd/',
                              phase='all_seq', cam=0, depth_mode='dror')
print('val or test data size:', len(data_gen_test.dataset))
dataloader_test = data_gen_test.create_data(batch_size=param['batch_size'])
data = next(iter(dataloader_test))
print('img shape:', data['img'].shape)
print('depth shape:', data['depth'].shape)

model = net.get_model(param['mode'], pretrained=False)
model.cuda()

utils.load_checkpoint(model_dir, model)
print('Load model parameters done')

pseudo_label(model, dataloader_test, param)
