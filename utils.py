from matplotlib import pyplot as plt
import os
import shutil
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')


def show_result(data, pred, aux_map, error_map, batch_size=10, shuffle=True):       # change to any list of maps [img, depth, pred, aux_map, error_map]
    if batch_size > 4:
        if shuffle:
            sample = np.arange(batch_size)
            np.random.shuffle(sample)
            sample = sample[:4]
        else:
            sample = list(range(4))
        if aux_map is None:
            fig, axes = plt.subplots(5, 4, figsize=(20, 10))
            for i, idx in enumerate(sample):
                axes[0, i].imshow(data['img'][idx].cpu().permute(1, 2, 0))
                axes[1, i].imshow(data['depth'][idx].cpu(), cmap='jet')
                axes[2, i].imshow(pred[idx], cmap='jet')
                axes[3, i].imshow(error_map[0][idx], cmap='viridis', vmin=0, vmax=80)
                axes[4, i].imshow(error_map[1][idx], cmap='viridis', vmin=0, vmax=10)
        else:
            fig, axes = plt.subplots(6, 4, figsize=(20, 12))
            for i, idx in enumerate(sample):
                axes[0, i].imshow(data['img'][idx].cpu().permute(1, 2, 0))
                axes[1, i].imshow(data['depth'][idx].cpu(), cmap='jet')
                axes[2, i].imshow(aux_map[idx], cmap='gray')
                axes[3, i].imshow(pred[idx], cmap='jet')
                axes[4, i].imshow(error_map[0][idx], cmap='viridis', vmin=0, vmax=80)
                axes[5, i].imshow(error_map[1][idx], cmap='viridis', vmin=0, vmax=10)

    elif 1 < batch_size <= 4:
        sample = list(range(batch_size))
        if aux_map is None:
            fig, axes = plt.subplots(5, batch_size, figsize=(batch_size * 5, 10))
            for i, idx in enumerate(sample):
                axes[0, i].imshow(data['img'][idx].cpu().permute(1, 2, 0))
                axes[1, i].imshow(data['depth'][idx].cpu(), cmap='jet')
                axes[2, i].imshow(pred[idx], cmap='jet')
                axes[3, i].imshow(error_map[0][idx], cmap='viridis', vmin=0, vmax=80)
                axes[4, i].imshow(error_map[1][idx], cmap='viridis', vmin=0, vmax=1)
        else:
            fig, axes = plt.subplots(6, batch_size, figsize=(batch_size * 5, 12))
            for i, idx in enumerate(sample):
                axes[0, i].imshow(data['img'][idx].cpu().permute(1, 2, 0))
                axes[1, i].imshow(data['depth'][idx].cpu(), cmap='jet')
                axes[2, i].imshow(aux_map[idx], cmap='gray')
                axes[3, i].imshow(pred[idx], cmap='jet')
                axes[4, i].imshow(error_map[0][idx], cmap='viridis', vmin=0, vmax=80)
                axes[5, i].imshow(error_map[1][idx], cmap='viridis', vmin=0, vmax=1)

    elif batch_size == 1:
        if aux_map is None:
            fig, axes = plt.subplots(5, 1, figsize=(5, 10))
            axes[0].imshow(data['img'][0].cpu().permute(1, 2, 0))
            axes[1].imshow(data['depth'][0].cpu(), cmap='jet')
            axes[2].imshow(pred[0], cmap='jet')
            axes[3].imshow(error_map[0][0], cmap='viridis', vmin=0, vmax=80)
            axes[4].imshow(error_map[1][0], cmap='viridis', vmin=0, vmax=1)
        else:
            fig, axes = plt.subplots(6, 1, figsize=(5, 12))
            axes[0].imshow(data['img'][0].cpu().permute(1, 2, 0))
            axes[1].imshow(data['depth'][0].cpu(), cmap='jet')
            axes[2].imshow(aux_map[0], cmap='gray')
            axes[3].imshow(pred[0], cmap='jet')
            axes[4].imshow(error_map[0][0], cmap='viridis', vmin=0, vmax=80)
            axes[5].imshow(error_map[1][0], cmap='viridis', vmin=0, vmax=1)

    return fig


def save_checkpoint(state, is_best, folder_path):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'

    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        folder_path: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(folder_path, 'last.pth.tar')
    if not os.path.exists(folder_path):
        print("Checkpoint Directory does not exist! Making directory {}".format(folder_path))
        os.mkdir(folder_path)
    else:
        print("Checkpoint Directory exists! ")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(folder_path, 'best.pth.tar'))


def load_checkpoint(file_path, model, optimizer=None, scheduler=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.

    Args:
        file_path: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(file_path):
        raise ("File doesn't exist {}".format(file_path))
    else:
        print('\nLoading parameters from {}'.format(file_path))

    checkpoint = torch.load(file_path)
    if 'state_dict' not in checkpoint.keys():
        checkpoint['state_dict'] = checkpoint.pop('model_state')

    pretrained_dict = checkpoint['state_dict']
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                       ((k in model_dict) and (v.size() == model_dict[k].size()))}
    print('Following model parameters are loaded:\n', pretrained_dict.keys(), '\n')
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)

    # model.load_state_dict(checkpoint['state_dict'], strict=False)

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    if (scheduler is not None) and ('sched_dict' in checkpoint):
        scheduler.load_state_dict(checkpoint['sched_dict'])

    return checkpoint


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


# def restore_states(file, model, optimizer=None, scheduler=None):
#     print("\nRestoring parameters from {}".format(file))
#     load_dict = load_checkpoint(file, model, optimizer, scheduler)
#     return load_dict

# if param['mode'] == checkpoint.get('mode', 'classification'):
#     load_dict = load_checkpoint(file, model, optimizer, scheduler)
#     return load_dict
#
# else:
#     if param['mode'] == 'classification':
#         model = net.get_model('regression')
#     elif param['mode'] == 'regression':
#         model = net.get_model('classification')
#     load_dict = load_checkpoint(file, model, optimizer, scheduler)
#     # Change last layer
#     in_channels = model.classifier[4].in_channels
#     kernel_size = model.classifier[4].kernel_size
#     if param['mode'] == 'classification':
#         model.classifier[4] = torch.nn.Conv2d(in_channels, net.K + 1, kernel_size)
#     elif param['mode'] == 'regression':
#         model.classifier[4] = torch.nn.Conv2d(in_channels, 1, kernel_size)
#     return load_dict


def freeze_classification(model):
    # Set ASPP layers requires_grad, backbone layers no need requires_grad
    for weight in model.parameters():
        weight.requires_grad = False
    for weight in model.reg_of_cls.parameters():
        weight.requires_grad = True
    print("Params to learn:")
    for name, weight in model.named_parameters():
        if weight.requires_grad:
            print("\t", name)
