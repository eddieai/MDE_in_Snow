import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
# torch.autograd.set_detect_anomaly(True)
# from kitti.dataset import DataGenerator
from cadc.dataset import DataGenerator
import net
import utils
from evaluate import evaluate
# from evaluate_pseudo import evaluate_pseudo


def train_pseudo(model, optimizer, dataloader, loss_fn, metric_fn, param, current_epoch):
    model.train()
    summary = []

    for i, data in enumerate(dataloader):
        img_batch = data['img']
        depth_dror = data['depth']
        pseudo_label = data['pseudo']
        # move to GPU if available
        img_batch = img_batch.cuda()

        # compute model output
        output_batch = model(img_batch)

        # clear previous gradients, compute loss
        optimizer.zero_grad()

        loss = loss_fn(output_batch.cpu(), depth_dror, pseudo_label, lamda=1).cuda()

        loss.backward()
        # performs updates using calculated gradients
        optimizer.step()

        if i % 10 == 0:
            print('Train: epoch %d iter %d loss: %.3f' % (current_epoch, i, loss))
            writer.add_scalar('training loss', loss, current_epoch * len(dataloader) + i)

            output_batch, depth_dror, pseudo_label = output_batch.cpu().detach().numpy(), depth_dror.numpy(), pseudo_label.numpy()
            pred_batch = net.depth_inference(output_batch, param['mode'])

            depth_batch = np.where(depth_dror != 0, depth_dror, pseudo_label)

            metrics = metric_fn(pred_batch, depth_batch)
            # for metric, value in metrics.items():
            #     print('Training epoch %d iter %d metric %s: %.3f' % (epoch, i, metric, value))
            #     writer.add_scalar(metric, value, epoch * len(dataloader) + i)
            metrics['loss'] = loss.item()
            summary.append(metrics)

        if i % 100 == 0:
            if param['mode'] in ['sord_ent_weighted', 'sord_weighted_minent']:
                aux_map = net.local_entropy(depth_batch, kernel=16, mask=True)
            elif param['mode'] == 'sord_min_local_ent':
                aux_map = net.local_entropy(pred_batch, kernel=16)
            elif param['mode'] == 'sord_align_grad':
                aux_map = net.edge(pred_batch)
            else:
                aux_map = None
            error_map = net.depth_error_map(pred_batch, depth_batch)
            show_result_fig = utils.show_result(data, pred_batch, aux_map, error_map, param['batch_size'], shuffle=True)
            writer.add_figure('Train  Input_RGB  Input_Depth  Output_Depth_map', show_result_fig, current_epoch * len(dataloader) + i)

    metrics_train = {metric: np.mean([x[metric] for x in summary]) for metric in summary[0]}
    print('\n------ After training %d epochs, train set metrics mean: ------ \n%s\n' % (current_epoch, metrics_train))


def train_evaluate(model, optimizer, scheduler, dataloader_train, dataloader_val, loss_fn, metric_fn, model_dir, param):
    best_SILog_val = float('inf')
    epoch_start = param['current_epoch']

    for epoch in range(epoch_start, param['epochs']):
        print()
        print('------ Epoch %d, Learning rate = %.2e ------' % (epoch, optimizer.param_groups[0]['lr']))
        train_pseudo(model, optimizer, dataloader_train, loss_fn, metric_fn, param, epoch)

        metrics_val = evaluate(model, dataloader_val, metric_fn, param, epoch, writer)
        # metrics_val = evaluate_pseudo(model, dataloader_val, loss_fn, metric_fn, param, epoch, writer)
        print(
            '\n------ After training %d epochs, validation set metrics mean: ------ \n%s\n' % (epoch, metrics_val))
        for metric, value in metrics_val.items():
            writer.add_scalar(metric, value, epoch)

        SILog_val = metrics_val['SILog']

        if scheduler is not None:
            scheduler.step(SILog_val)

        is_best = SILog_val <= best_SILog_val

        # Save weights
        save_dict = param.copy()
        save_dict['current_epoch'] = epoch
        save_dict['state_dict'] = model.state_dict()
        save_dict['optim_dict'] = optimizer.state_dict()
        if scheduler is not None:
            save_dict['sched_dict'] = scheduler.state_dict()
        experiments_dir = 'experiments/' + model_dir
        utils.save_checkpoint(save_dict, is_best=is_best, folder_path=experiments_dir)

        # If best_eval, best_save_path
        if is_best:
            best_SILog_val = SILog_val

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(experiments_dir, "metrics_test_best_weights.json")
            utils.save_dict_to_json(metrics_val, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(experiments_dir, "metrics_test_last_weights.json")
        utils.save_dict_to_json(metrics_val, last_json_path)


if __name__ == '__main__':
    param = {
        'phase': 'train',
        'batch_size': 12,   # 3
        'eval_n_crop': 4,
        'learning_rate': 1e-3,
        'momentum': 0.9,
        'weight_decay': 0,
        'epochs': 40,
        'mode': 'sord'   # sord, sord_ent_weighted, sord_min_local_ent, sord_weighted_minent, sord_align_grad, classification, regression, reg_of_cls
    }
    train_type = 'refine'  # refine or continue
    model_dir = 'refine_CADC_rescaled_seq_pseudo_label_depth_dror_val_dror_lr_1e-03_momentum_0.9_wd_0_epoch_30_mode_sord_pretrained_DeepLabV3+_Kitti'

    if train_type == 'refine':
        restore_file = 'experiments/train_lr_1e-03_momentum_0.9_wd_0_epoch_30_mode_sord_pretrained_DeepLabV3+_PascalVOC_crop_375*513/best.pth.tar'
    else:
        restore_file = 'experiments/%s/last.pth.tar' % model_dir

    model = net.get_model(param['mode'], pretrained=False)
    model.cuda()

    ## Pretrained on Pascal semantic segmentation, retrain on Kitti depth estimation
    # last_layer = ['classifier.classifier.4.weight', 'classifier.classifier.4.bias']
    # last_layer_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in last_layer, model.named_parameters()))))
    # base_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] not in last_layer, model.named_parameters()))))
    #
    # optimizer = torch.optim.SGD([
    #     {'params': base_params, 'lr': param['learning_rate']},
    #     {'params': last_layer_params, 'lr': param['learning_rate'] * 10}
    # ], momentum=param['momentum'], weight_decay=param['weight_decay'], nesterov=True)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1)

    optimizer = torch.optim.SGD(params=model.parameters(), lr=param['learning_rate'], momentum=param['momentum'], weight_decay=param['weight_decay'], nesterov=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)

    param['current_epoch'] = 0
    if restore_file is not None:
        if train_type == 'refine':
            load_dict = utils.load_checkpoint(restore_file, model, optimizer=None, scheduler=None)
        elif train_type == 'continue':
            load_dict = utils.load_checkpoint(restore_file, model, optimizer=optimizer, scheduler=scheduler)
            assert 'current_epoch' in load_dict, "current_epoch does not exist in restore file"
            param['current_epoch'] = load_dict['current_epoch']+1

    # if load model of "sord_ent_weighted" before 08/11, need to manually set model to "sord_ent_weighted"
    # param['mode'] = 'sord_ent_weighted'

    if param['mode'] == 'reg_of_cls':
        utils.freeze_classification(model)

    if model_dir is None:
        if restore_file is not None:
            model_dir = restore_file.split('/')[1].strip()
        else:
            model_dir = '%s_lr_%.2e_wd_%.2e_epoch_%d_mode_%s' % (param['phase'], param['learning_rate'], param['weight_decay'], param['epochs'], param['mode'])

    writer = SummaryWriter('runs/' + model_dir)

    # data_gen_train = DataGenerator('/home/datasets/Kitti/', phase=param['phase'])
    data_gen_train = DataGenerator('/home/datasets/CADC/cadcd/', '/home/datasets_mod/CADC/cadcd/',
                                   phase='train_seq', cam=0, depth_mode='pseudo_dror', rescaled=True)
    print('train data size:', len(data_gen_train.dataset))
    dataloader_train = data_gen_train.create_data(batch_size=param['batch_size'])
    data = next(iter(dataloader_train))
    print('img shape:', data['img'].shape)
    print('depth shape:', data['depth'].shape)

    # data_gen_val = DataGenerator('/home/datasets/Kitti/', phase='test')
    data_gen_val = DataGenerator('/home/datasets/CADC/cadcd/', '/home/datasets_mod/CADC/cadcd/',
                                 phase='val_seq', cam=0, depth_mode='dror', rescaled=True)
    print('val data size:', len(data_gen_val.dataset))
    dataloader_val = data_gen_val.create_data(batch_size=param['batch_size'])

    # # Eval on both current corresponding snowfall val set & all CADC val set
    # data_gen_val_current = DataGenerator('/home/datasets/CADC/cadcd/', '/home/datasets_mod/CADC/cadcd/', phase='val', dror=True, cam=0, snow_level='light')
    # print('current snowfall\'s val data size:', len(data_gen_val_current.dataset))
    # dataloader_val_current = data_gen_val_current.create_data(batch_size=param['batch_size'])
    #
    # data_gen_val_all = DataGenerator('/home/datasets/CADC/cadcd/', '/home/datasets_mod/CADC/cadcd/', phase='val', dror=True, cam=0, snow_level=None)
    # print('all CADC val data size:', len(data_gen_val_all.dataset))
    # dataloader_val_all = data_gen_val_all.create_data(batch_size=param['batch_size'])
    #
    # dataloader_val = {'current': dataloader_val_current, 'all': dataloader_val_all}

    train_evaluate(model, optimizer, scheduler, dataloader_train, dataloader_val, net.loss_fn_pseudo, net.metric_fn, model_dir, param)
