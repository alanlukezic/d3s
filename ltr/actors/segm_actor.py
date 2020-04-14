from . import BaseActor
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os


def draw_axis(ax, img, title, show_minmax=False):
    ax.imshow(img)
    if show_minmax:
        minval_, maxval_, _, _ = cv2.minMaxLoc(img)
        title = '%s \n min=%.2f max=%.2f' % (title, minval_, maxval_)
    ax.set_title(title, fontsize=9)



def save_debug(data, pred_mask):
    batch_element = 0
    dir_path = data['settings'].env.images_dir

    train_img = data['train_images'][:, batch_element, :, :].permute(1, 2, 0)
    test_img = data['test_images'][:, batch_element, :, :].permute(1, 2, 0)
    test_mask = data['test_masks'][0, batch_element, :, :]

    # softmax on the mask prediction (since this is done internaly when calculating loss)
    mask = F.softmax(pred_mask, dim=1)[batch_element, 0, :, :]


    mu = torch.Tensor(data['settings'].normalize_mean).to(torch.device('cuda')).view(1, 1, 3)
    std = torch.Tensor(data['settings'].normalize_std).to(torch.device('cuda')).view(1, 1, 3)

    train_img = 255 * (train_img * std + mu)
    test_img = 255 * (test_img * std + mu)

    train_img = (train_img.cpu().numpy()).astype(np.uint8)
    test_img = (test_img.cpu().numpy()).astype(np.uint8)
    test_mask = (test_mask.cpu().numpy()).astype(np.float32)
    predicted_mask = (mask.cpu().detach().numpy()).astype(np.float32)

    f, ((ax1, ax2), (ax5, ax6)) = plt.subplots(2, 2, figsize=(6, 6))
    draw_axis(ax1, train_img, 'Train image')
    draw_axis(ax5, test_img, 'Test image')
    draw_axis(ax2, test_mask, 'Ground-truth')
    draw_axis(ax6, predicted_mask, 'Prediction', show_minmax=True)

    save_path = os.path.join(data['settings'].env.images_dir, '%03d-%04d.png' % (data['epoch'], data['iter']))
    plt.savefig(save_path)
    plt.close(f)


class SegmActor(BaseActor):
    """ Actor for training the Segmentation in ATOM"""
    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals' and 'proposal_iou'.

        returns:
            loss    - the training loss
            states  -  dict containing detailed losses
        """

        test_dist = None
        if 'test_dist' in data:
            test_dist = data['test_dist'].permute(1, 0, 2, 3)

        # Run network to obtain IoU prediction for each proposal in 'test_proposals'
        masks_pred = self.net(data['train_images'].permute(1, 0, 2, 3),
                              data['test_images'].permute(1, 0, 2, 3),
                              data['train_masks'].permute(1, 0, 2, 3),
                              test_dist)

        masks_gt = data['test_masks'].permute(1, 0, 2, 3)
        masks_gt_pair = torch.cat((masks_gt, 1 - masks_gt), dim=1)

        # Compute loss
        loss = self.objective(masks_pred, masks_gt_pair)

        # Return training stats
        stats = {'Loss/total': loss.item(),
                 'Loss/segm': loss.item()}

        if 'iter' in data and (data['iter'] - 1) % 50 == 0:
            save_debug(data, masks_pred)

        return loss, stats