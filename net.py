import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import network
from skimage.filters.rank import entropy
from scipy.stats import entropy as scipy_entropy
from scipy.special import softmax as scipy_softmax
import cv2
import matplotlib
matplotlib.use('Agg')


np.seterr(divide='ignore', invalid='ignore')

a = 1
b = 80
K = 120
bins_type = 'log'       # log, linear

if bins_type == 'linear':
    bins = [a + (b - a) * i / K for i in range(K)]
elif bins_type == 'log':
    bins = [np.exp(np.log(a) + np.log(b / a) * i / K) for i in range(K)]
bins_all = np.array([0] + bins + [80])
center = (bins_all[1:] + bins_all[:-1]) / 2


VISUALIZE_sord = False                  # Must set to False when real training!!!
VISUALIZE_sord_ent_weighted = False     # Must set to False when real training!!!
VISUALIZE_sord_min_local_ent = False    # Must set to False when real training!!!
VISUALIZE_sord_align_grad = False       # Must set to False when real training!!!

if VISUALIZE_sord or VISUALIZE_sord_ent_weighted or VISUALIZE_sord_min_local_ent or VISUALIZE_sord_align_grad:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    matplotlib.use('TkAgg')
    plt.ion()
    cmap = cm.get_cmap('jet')


def get_model(mode, pretrained=False):
    if mode in ['classification', 'reg_of_cls'] or mode[:4] == 'sord':
        return network.deeplabv3plus_resnet101(num_classes=K + 1, output_stride=8, pretrained_backbone=pretrained)

    # assert mode in ['regression', 'classification', 'reg_of_cls', 'sord'], 'mode must be regression or classification or reg_of_cls'
    #
    # # depth classification or multi_task model
    # if mode in ['classification', 'reg_of_cls', 'sord']:
    #     return deeplab.deeplabv3_resnet101(mode, num_classes=K+1, pretrained=pretrained)
    #     # return deeplabv3_resnet101(pretrained=False, progress=True, num_classes=K+1, aux_loss=None)
    #
    # # depth regression model
    # if mode == 'regression':
    #     return deeplab.deeplabv3_resnet101(mode, pretrained=pretrained)
    #     # return deeplabv3_resnet101(pretrained=False, progress=True, num_classes=1, aux_loss=None)


def local_entropy(depth, kernel=16, mask=False):
    depth_entropy_sequence = []

    for i in range(depth.shape[0]):
        depth_i = depth[i].clip(0, b).astype(np.uint8)

        # compute entropy directly on depth, entropy kernel 16x16
        if mask:
            depth_entropy_i = entropy(depth_i, selem=np.ones((kernel, kernel)).astype(np.uint8), mask=(depth_i > 0))
        else:
            depth_entropy_i = entropy(depth_i, selem=np.ones((kernel, kernel)).astype(np.uint8))

        depth_entropy_sequence.append(depth_entropy_i)
    return np.stack(depth_entropy_sequence, axis=0)


def edge(img):
    edge_sequence = []

    for i in range(img.shape[0]):
        if len(img[i].shape) == 3:
            img_grayscale = cv2.cvtColor(np.uint8(img[i]*255).transpose(1, 2, 0)[..., ::-1], cv2.COLOR_BGR2GRAY)
            # img_edge = cv2.Canny(img_grayscale, 10, 50)
        elif len(img[i].shape) == 2:
            img_grayscale = np.uint8(img[i]/80*255)
            # img_edge = cv2.Canny(img_grayscale, 10, 50)

        # Sobel
        sobelx = cv2.Sobel(img_grayscale, cv2.CV_64F, 1, 0)
        sobely = cv2.Sobel(img_grayscale, cv2.CV_64F, 0, 1)
        sobelx = cv2.convertScaleAbs(sobelx)
        sobely = cv2.convertScaleAbs(sobely)
        img_edge = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)

        ## Scharr
        # scharrx = cv2.Scharr(img_grayscale, cv2.CV_64F, 1, 0)
        # scharry = cv2.Scharr(img_grayscale, cv2.CV_64F, 0, 1)
        # scharrx = cv2.convertScaleAbs(scharrx)
        # scharry = cv2.convertScaleAbs(scharry)
        # img_edge = cv2.addWeighted(scharrx, 0.5, scharry, 0.5, 0)

        ## Laplacian
        # laplacian = cv2.Laplacian(img_grayscale, cv2.CV_64F)
        # img_edge = cv2.convertScaleAbs(laplacian)

        ## Canny
        # img_edge = cv2.Canny(img_grayscale, 50, 100)

        edge_sequence.append(img_edge)

    return np.stack(edge_sequence, axis=0)


def depth_inference(output, mode):
    if mode == 'classification':
        pred_class = np.argmax(output, axis=1)
        pred_map = center[pred_class.reshape(-1)].reshape(*pred_class.shape)
    elif mode in ['regression', 'reg_of_cls']:
        pred_map = output.clip(1e-3, b).squeeze()
    elif mode[:4] == 'sord':
        pred_map = np.sum(F.softmax(torch.tensor(output), dim=1).numpy() * np.array(center).reshape(1, -1, 1, 1),
                          axis=1)
        # pred_class = np.argmax(output, axis=1)
        # pred_map = center[pred_class.reshape(-1)].reshape(*pred_class.shape)
    return pred_map


def depth_error_map(pred, depth):
    gt = depth.clip(0, b)
    absolute_error = np.where(gt != 0, np.abs(pred - gt), 0)
    absolute_error_maxpool = F.max_pool2d(torch.tensor(absolute_error[:, None, :, :]), kernel_size=4)
    absolute_error_interpolate = F.interpolate(absolute_error_maxpool, size=depth.shape[-2:]).squeeze().numpy()
    relative_error = np.where(gt != 0, absolute_error / gt, 0)
    relative_error_maxpool = F.max_pool2d(torch.tensor(relative_error[:, None, :, :]), kernel_size=4)
    relative_error_interpolate = F.interpolate(relative_error_maxpool, size=depth.shape[-2:]).squeeze().numpy()
    return [absolute_error_interpolate, relative_error_interpolate]


def loss_fn(output, depth, mode, img=None):
    if mode == 'classification':
        depth_np = depth.cpu().numpy()
        label = np.digitize(np.clip(depth_np, 0, b), bins)
        label = torch.Tensor(label).long().cuda()

        C = output.size()[1]
        mask = (depth != 0.).cuda()
        mask = mask[:, None, :, :].repeat_interleave(C, dim=1)

        h = torch.arange(0., C).view(1, -1).cuda()
        information_gain = torch.exp(-0.5 * (h - h.T) ** 2)
        H = information_gain[label.view(-1), :].view(*label.size(), C).cuda()
        H = H.permute(0, 3, 1, 2)[mask]

        P = F.log_softmax(output, dim=1)[mask]

        return - torch.mean((H * P)) * C

    if mode in ['regression', 'reg_of_cls']:
        mask = (depth != 0).cuda()
        gt_mask = depth.clamp(0, b)[mask]
        pred_mask = output.clamp(1e-3, b).squeeze()[mask]

        # dlog = torch.log(gt_mask) - torch.log(pred_mask)
        # loss = torch.mean(dlog ** 2) - torch.mean(dlog) ** 2
        if mode == 'regression':
            criterion = nn.MSELoss()
            return criterion(pred_mask, gt_mask)
        if mode == 'reg_of_cls':
            return torch.mean(torch.log(torch.cosh(pred_mask - gt_mask + 1e-12)))

    if mode[:4] == 'sord':
        mask = (depth != 0)
        gt = depth.clamp(0, b)[:, :, :, None]
        if mode == 'sord_logphi':
            phi = (torch.log(gt) - torch.log(torch.tensor(center).float()).view(1,1,1,-1))**2
        else:
            phi = (gt - torch.tensor(center).float().view(1, 1, 1, -1)) ** 2
        gt_sord = F.softmax(-phi, dim=3)[mask]
        log_p = F.log_softmax(output, dim=1).permute(0, 2, 3, 1)[mask]

        if VISUALIZE_sord:
            # Plot Ground-truth SORD of a pixel and Output of the same pixel
            # fig1 = plt.figure(0)
            # plt.cla()
            # axes = fig1.subplots(1, 2, sharex=True, sharey=True)
            # axes[0].bar(center, gt_sord.detach().cpu().numpy()[0], width=np.diff(np.append(center, [80])) / 2,
            #             align='edge', color=cmap(np.arange(K).astype(float) / K))
            # axes[0].set_title('Ground-truth SORD of a pixel')
            # axes[0].set_facecolor('black')
            # axes[0].set_xscale('log')
            # axes[0].set_xlim(0.5, 80)
            # axes[0].set_ylim(0, 1)
            # axes[0].xaxis.set_minor_locator(ticker.FixedLocator([1] + list(range(10, 81, 10))))
            # axes[0].xaxis.set_major_locator(ticker.NullLocator())
            # axes[0].xaxis.set_minor_formatter(ticker.ScalarFormatter())
            # axes[1].bar(center, np.exp(log_p.detach().cpu().numpy())[0], width=np.diff(np.append(center, [80])) / 2,
            #             align='edge', color=cmap(np.arange(K).astype(float) / K))
            # axes[1].set_title('Output of the same pixel')
            # axes[1].set_facecolor('black')

            log_p_unmask = F.log_softmax(output, dim=1).permute(0, 2, 3, 1)
            p_unmask = F.softmax(output, dim=1).permute(0, 2, 3, 1)
            E = - torch.sum((p_unmask * log_p_unmask), dim=3)
            # output_softmax = scipy_softmax(output.detach().cpu().numpy(), axis=1)
            # output_softmax = output_softmax.transpose(0,2,3,1)
            # E = scipy_entropy(output_softmax, axis=3)
            pred_map = depth_inference(output.detach().cpu().numpy(), mode=mode)

            fig2 = plt.figure(1)
            plt.cla()
            axes = fig2.subplots(2, 2)
            axes[0,0].imshow(img[0].cpu().permute(1, 2, 0))
            axes[0,0].set_title('RGB image')
            axes[0,1].imshow(depth[0].cpu().numpy(), cmap='jet')
            axes[0,1].set_title('depth map')
            axes[1,0].imshow(pred_map[0], cmap='jet')
            axes[1,0].set_title('predicted depth map')
            axes[1,1].imshow(E[0].detach().cpu().numpy(), cmap='viridis')
            axes[1,1].set_title('pixel-wise entropy of predicted')
            plt.pause(0.1)
            plt.show()

        if mode in ['sord', 'sord_logphi']:
            # Normal KLDivergence loss
            criterion = nn.KLDivLoss(reduction='batchmean')
            return criterion(log_p, gt_sord)

        elif mode == 'sord_ent_weighted':
            # KLDivergence loss weighted according to ground truth depthmap local entropy
            ## entropy kernel 16x16
            gt_entropy = torch.Tensor(local_entropy(depth.cpu().numpy(), kernel=16, mask=True))

            if VISUALIZE_sord_ent_weighted:
                for i in range(output.size()[0]):
                    fig = plt.figure(i)
                    plt.cla()
                    plt.axis('off')
                    axes = fig.subplots(1, 3, sharex=True, sharey=True)
                    axes[0].imshow(img[i].cpu().permute(1, 2, 0))
                    axes[0].set_title('RGB image')
                    axes[1].imshow(depth[i].cpu().numpy(), cmap='jet')
                    axes[1].set_title('Depth map (ground truth)')
                    axes[2].imshow(gt_entropy[i].cpu().numpy(), cmap='gray')
                    axes[2].set_title('Depth map Entropy')
                    plt.pause(0.1)
                    plt.show()

            gt_entropy_mask = gt_entropy[mask]
            ## linear
            # weight_by_entropy = torch.clamp(1 - gt_entropy_mask / 6, min=0)         ## entropy kernel 16x16, divide by 6; entropy kernel 3x3, divide by 3
            # weight_by_entropy = 1 - gt_entropy_mask / gt_entropy_mask.max()
            ## sigmoid
            weight_by_entropy = 1 - F.sigmoid(gt_entropy_mask)

            KLDiv = torch.sum(F.kl_div(log_p, gt_sord, reduction='none'), dim=1)
            return torch.sum(KLDiv * weight_by_entropy) / torch.sum(weight_by_entropy)
        #
        # elif mode == 'sord_min_local_ent':
        #     # Normal KLDivergence loss
        #     criterion = nn.KLDivLoss(reduction='batchmean')
        #     loss_sord = criterion(log_p, gt_sord)
        #
        #     pred_map = depth_inference(output.detach().cpu().numpy(), mode=mode)
        #
        #     # entropy of masked predicted depth map (could be wrong)
        #     pred_entropy = torch.Tensor(local_entropy(pred_map, kernel=16)).cuda()
        #
        #     if VISUALIZE_sord_min_local_ent:
        #         for i in range(output.size()[0]):
        #             fig = plt.figure(i)
        #             plt.cla()
        #             plt.axis('off')
        #             axes = fig.subplots(2, 2, sharex=True, sharey=True)
        #             axes[0, 0].imshow(img[i].cpu().permute(1, 2, 0))
        #             axes[0, 0].set_title('RGB image')
        #             axes[0, 1].imshow(depth[i].cpu().numpy(), cmap='jet')
        #             axes[0, 1].set_title('Depth map (ground truth)')
        #             axes[1, 0].imshow(pred_map[i], cmap='jet')
        #             axes[1, 0].set_title('Predicted depth map')
        #             axes[1, 1].imshow(pred_entropy[i].cpu().numpy(), cmap='gray')
        #             axes[1, 1].set_title('Predicted depth map Entropy')
        #             plt.pause(0.1)
        #             plt.show()
        #
        #     loss_min_ent = torch.mean(pred_entropy)
        #     print('loss_sord', loss_sord, 'loss_min_ent', loss_min_ent)
        #     # return loss_sord + 0.1 * loss_min_ent
        #     return loss_sord + 1 * loss_min_ent
        #     # return loss_sord + F.sigmoid(loss_min_ent) * 2 - 1 ?
        #
        # elif mode == 'sord_weighted_minent':
        #     # compute loss_kl_weighted
        #     gt_entropy = torch.Tensor(local_entropy(depth.cpu().numpy(), kernel=16, mask=True)).cuda()
        #     gt_entropy_mask = gt_entropy[mask]
        #
        #     ## linear
        #     # weight_by_entropy = torch.clamp(1 - gt_entropy_mask / 6, min=0)         ## entropy kernel 16x16, divide by 6; entropy kernel 3x3, divide by 3
        #     # weight_by_entropy = 1 - gt_entropy_mask / gt_entropy_mask.max()
        #     ## sigmoid
        #     weight_by_entropy = 1 - F.sigmoid(gt_entropy_mask)
        #
        #     KLDiv = torch.sum(F.kl_div(log_p, gt_sord, reduction='none'), dim=1)
        #     loss_sord_weighted = torch.sum(KLDiv * weight_by_entropy) / torch.sum(weight_by_entropy)
        #
        #     # compute loss_minEnt
        #     pred_map = depth_inference(output.detach().cpu().numpy(), mode=mode)
        #     # entropy of masked predicted depth map (could be wrong)
        #     pred_entropy = torch.Tensor(local_entropy(pred_map, kernel=16)).cuda()
        #     loss_min_ent = torch.mean(pred_entropy)
        #
        #     print('loss_sord_weighted', loss_sord_weighted, 'loss_min_ent', loss_min_ent)
        #     return loss_sord_weighted + 1 * loss_min_ent
        #
        # elif mode == 'sord_align_grad':
        #     assert img is not None
        #
        #     # Normal KLDivergence loss
        #     criterion = nn.KLDivLoss(reduction='batchmean')
        #     loss_sord = criterion(log_p, gt_sord)
        #
        #     pred_map = depth_inference(output.detach().cpu().numpy(), mode=mode)
        #
        #     img_edge = torch.Tensor(edge(img.cpu().numpy())).cuda()
        #     pred_edge = torch.Tensor(edge(pred_map)).cuda()
        #
        #     if VISUALIZE_sord_align_grad:
        #         for i in range(output.size()[0]):
        #             fig = plt.figure(i)
        #             plt.cla()
        #             plt.axis('off')
        #             axes = fig.subplots(2, 2, sharex=True, sharey=True)
        #             axes[0, 0].imshow(img[i].cpu().permute(1, 2, 0))
        #             axes[0, 0].set_title('RGB image')
        #             axes[0, 1].imshow(img_edge[i].cpu().numpy(), cmap='gray')
        #             axes[0, 1].set_title('RGB Edge')
        #             axes[1, 0].imshow(pred_map[i], cmap='jet')
        #             axes[1, 0].set_title('Predicted depth map')
        #             axes[1, 1].imshow(pred_edge[i].cpu().numpy(), cmap='gray')
        #             axes[1, 1].set_title('Predicted depth map Edge')
        #             plt.pause(0.1)
        #             plt.show()
        #
        #     ## loss mean absolute error
        #     loss_align_grad = torch.mean(torch.abs(img_edge - pred_edge))
        #     ## loss KLDivergence
        #     # loss_align_grad = F.kl_div(img_edge, pred_edge) ?
        #
        #     ## with mask
        #     # img_edge_mask = img_edge[mask]
        #     # pred_edge_mask = pred_edge[mask]
        #     # loss_align_grad = torch.mean(torch.abs(img_edge_mask - pred_edge_mask))
        #     print('loss_sord', loss_sord, 'loss_align_grad', loss_align_grad)
        #     return loss_sord + 0.1 * loss_align_grad
        #     # return loss_sord + F.sigmoid(loss_align_grad) * 2 - 1 ?


def loss_fn_pseudo(output, depth_dror, pseudo_label, lamda=1):
    mask_dror = (depth_dror != 0)
    gt_dror = depth_dror.clamp(0, b)[:, :, :, None]
    phi_dror = (gt_dror - torch.tensor(center).float().view(1, 1, 1, -1)) ** 2
    gt_sord_dror = F.softmax(-phi_dror, dim=3)[mask_dror]
    log_p_dror = F.log_softmax(output, dim=1).permute(0, 2, 3, 1)[mask_dror]

    pseudo_label_no_dror = torch.where(depth_dror != 0, torch.zeros_like(depth_dror), pseudo_label)
    mask_pseudo = (pseudo_label_no_dror != 0)
    gt_pseudo = pseudo_label_no_dror.clamp(0, b)[:, :, :, None]
    phi_pseudo = (gt_pseudo - torch.tensor(center).float().view(1, 1, 1, -1)) ** 2
    gt_sord_pseudo = F.softmax(-phi_pseudo, dim=3)[mask_pseudo]
    log_p_pseudo = F.log_softmax(output, dim=1).permute(0, 2, 3, 1)[mask_pseudo]

    # Normal KLDivergence loss
    criterion = nn.KLDivLoss(reduction='batchmean')
    loss_dror = criterion(log_p_dror, gt_sord_dror)
    loss_pseudo = criterion(log_p_pseudo, gt_sord_pseudo)
    print('loss_dror = %.3f' % loss_dror.item(), '\tloss_pseudo = %.3f' % loss_pseudo.item())

    return loss_dror + lamda * loss_pseudo


def metric_fn(pred, depth):
    mask = (depth != 0)
    gt_mask = depth.clip(0, b)[mask]
    pred_mask = pred[mask]

    d = gt_mask - pred_mask
    dlog = np.log(gt_mask) - np.log(pred_mask)
    dmax = np.maximum(gt_mask / pred_mask, pred_mask / gt_mask)

    metrics = {}
    metrics['SILog'] = np.sqrt((np.mean(dlog ** 2) - np.mean(dlog) ** 2))
    metrics['sqErrorRel'] = np.mean(d ** 2 / gt_mask ** 2)
    metrics['absErrorRel'] = np.mean((np.abs(d) / gt_mask))
    metrics['iRMSE'] = np.sqrt(np.mean((1 / gt_mask - 1 / pred_mask) ** 2))
    metrics['RMSE'] = np.sqrt(np.mean(d ** 2))
    metrics['RMSELog'] = np.sqrt(np.mean(dlog ** 2))
    metrics['accuracy_thres_%'] = np.mean(dmax < 1.25) * 100
    metrics['accuracy_thres**2_%'] = np.mean(dmax < 1.25 ** 2) * 100
    metrics['accuracy_thres**3_%'] = np.mean(dmax < 1.25 ** 3) * 100

    return metrics


def equally_spaced_crop(img_batch, n_crop):
    W = img_batch.size()[-1]
    cropped = []
    equal_space = round((W - 513) / (n_crop - 1))
    for n in range(n_crop - 1):
        left = n * equal_space
        cropped.append(img_batch[..., left:left + 513])
    cropped.append(img_batch[..., -513:])
    return torch.cat(cropped, dim=0)


def pred_overlap(pred_cropped, target_shape, n_crop):
    N, H, W = target_shape
    pred = np.zeros((N, H, W))
    overlap = np.zeros((H, W))
    equal_space = round((W - 513) / (n_crop - 1))

    for n in range(n_crop - 1):
        left = n * equal_space
        overlap[:, left:left + 513] += 1
        pred[..., left:left + 513] += pred_cropped[n * N:(n + 1) * N, ...]
    overlap[..., -513:] += 1
    pred[..., -513:] += pred_cropped[-N:, ...]

    pred /= overlap
    return pred
