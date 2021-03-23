import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
# from kitti.dataset import DataGenerator
from cadc.dataset import DataGenerator
import net
import utils


def evaluate(model, dataloader, metric_fn, param, current_epoch, writer):
    model.eval()
    summary = []

    for i, data in enumerate(dataloader):

        img_batch = data['img']
        depth_batch = data['depth']
        # move to GPU if available
        img_batch_cropped = net.equally_spaced_crop(img_batch, param['eval_n_crop'])
        depth_batch_cropped = net.equally_spaced_crop(depth_batch,  param['eval_n_crop'])
        img_batch_cropped, depth_batch_cropped = img_batch_cropped.cuda(), depth_batch_cropped.cuda()

        with torch.no_grad():
            # compute model output
            output_batch_cropped = model(img_batch_cropped)
            # compute loss (No need to compute loss for evaluation, save time)
            # loss = loss_fn(output_batch_cropped, depth_batch_cropped, param['mode'], img=img_batch_cropped)

        output_batch_cropped, depth_batch = output_batch_cropped.cpu().numpy(), depth_batch.cpu().numpy()
        pred_batch_cropped = net.depth_inference(output_batch_cropped, param['mode'])
        pred_batch = net.pred_overlap(pred_batch_cropped, depth_batch.shape, param['eval_n_crop'])
        metrics = metric_fn(pred_batch, depth_batch)
        # metrics['loss'] = loss.item()
        summary.append(metrics)

        if i % 10 == 0:
            print('------')
            for metric, value in metrics.items():
                print('Val: epoch %d iter %d metric %s: %.3f' % (current_epoch, i, metric, value))
            if param['mode'] in ['sord_ent_weighted', 'sord_weighted_minent']:
                aux_map = net.local_entropy(depth_batch, kernel=16, mask=True)
            elif param['mode'] == 'sord_min_local_ent':
                aux_map = net.local_entropy(pred_batch, kernel=16)
            elif param['mode'] == 'sord_align_grad':
                aux_map = net.edge(pred_batch)
            else:
                aux_map = None
            error_map = net.depth_error_map(pred_batch, depth_batch)
            show_result_fig = utils.show_result(data, pred_batch, aux_map, error_map, param['batch_size'], shuffle=False)
            writer.add_figure('Val  Input_RGB  Input_Depth  Output_Depth_map', show_result_fig, current_epoch * len(dataloader) + i)

            # for metric, value in metrics.items():
            #     writer.add_scalar(metric, value, i)

    metrics_mean = {metric: np.mean([x[metric] for x in summary]) for metric in summary[0]}
    return metrics_mean


if __name__ == '__main__':
    model_dir = "experiments/refine2_CADC_seq_depth_aggregated_corrected_HPR_ProjectedKNN_lr_1e-03_momentum_0.9_wd_0_epoch_30_mode_sord_pretrained_DeepLabV3+_Kitti/best.pth.tar"

    param = torch.load(model_dir).copy()
    # param['eval_n_crop'] = 4
    # param['batch_size'] = 3
    param.pop('state_dict')
    param.pop('optim_dict')
    param.pop('sched_dict', None)
    param.pop('restore_file', None)
    param.pop('model_dir', None)

    # writer = SummaryWriter('runs/' + model_dir.split('/')[1].strip().replace('train', 'test'))
    # writer = SummaryWriter('runs/test_lr_%.2e_wd_%.2e_epoch_%d_mode_%s' % (param['learning_rate'], param['weight_decay'], param['epochs'], param['mode']))
    writer = SummaryWriter('runs/test_aggregated_3_refine2_CADC_seq_depth_aggregated_corrected_HPR_ProjectedKNN_lr_1e-03_momentum_0.9_wd_0_epoch_30_mode_sord_pretrained_DeepLabV3+_Kitti/')

    # data_gen_test = DataGenerator('/home/datasets/Kitti/', phase='test')
    data_gen_test = DataGenerator('/home/datasets/CADC/cadcd/', '/home/datasets_mod/CADC/cadcd/', phase='test_seq',
                                  depth_mode='aggregated_3', cam=0)
    print('val or test data size:', len(data_gen_test.dataset))
    dataloader_test = data_gen_test.create_data(batch_size=param['batch_size'])
    data = next(iter(dataloader_test))
    print('img shape:', data['img'].shape)
    print('depth shape:', data['depth'].shape)

    model = net.get_model(param['mode'], pretrained=False)
    model.cuda()

    utils.load_checkpoint(model_dir, model)
    print('Load model parameters done')

    metrics_mean = evaluate(model, dataloader_test, net.metric_fn, param, param['current_epoch'], writer)
    print('------ Test set metrics mean: ------ \n%s\n' % metrics_mean)

    # save_path = os.path.join(os.path.dirname(model_dir), 'metrics_test.json')
    save_path = os.path.join(os.path.dirname(model_dir), 'test_seq_depth_aggregated_3.json')
    utils.save_dict_to_json(metrics_mean, save_path)
    print('Save metrics of test set done')
