import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))

def conv_no_relu(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.BatchNorm2d(out_planes))


def valid_roi(roi: torch.Tensor, image_size: torch.Tensor):
    valid = all(0 <= roi[:, 1]) and all(0 <= roi[:, 2]) and all(roi[:, 3] <= image_size[0]-1) and \
            all(roi[:, 4] <= image_size[1]-1)
    return valid


def normalize_vis_img(x):
    x = x - np.min(x)
    x = x / np.max(x)
    return (x * 255).astype(np.uint8)


class SegmNet(nn.Module):
    """ Network module for IoU prediction. Refer to the paper for an illustration of the architecture."""
    def __init__(self, segm_input_dim=(128,256), segm_inter_dim=(256,256), segm_dim=(64, 64), mixer_channels=2, topk_pos=3, topk_neg=3):
        super().__init__()

        self.segment0 = conv(segm_input_dim[3], segm_dim[0], kernel_size=1, padding=0)
        self.segment1 = conv_no_relu(segm_dim[0], segm_dim[1])

        self.mixer = conv(mixer_channels, segm_inter_dim[3])
        self.s3 = conv(segm_inter_dim[3], segm_inter_dim[2])

        self.s2 = conv(segm_inter_dim[2], segm_inter_dim[2])
        self.s1 = conv(segm_inter_dim[1], segm_inter_dim[1])
        self.s0 = conv(segm_inter_dim[0], segm_inter_dim[0])

        self.f2 = conv(segm_input_dim[2], segm_inter_dim[2])
        self.f1 = conv(segm_input_dim[1], segm_inter_dim[1])
        self.f0 = conv(segm_input_dim[0], segm_inter_dim[0])

        self.post2 = conv(segm_inter_dim[2], segm_inter_dim[1])
        self.post1 = conv(segm_inter_dim[1], segm_inter_dim[0])
        self.post0 = conv_no_relu(segm_inter_dim[0], 2)

        # Init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

        self.topk_pos = topk_pos
        self.topk_neg = topk_neg

    def forward(self, feat_test, feat_train, mask_train, test_dist=None):
        f_test = self.segment1(self.segment0(feat_test[3]))
        f_train = self.segment1(self.segment0(feat_train[3]))

        # reshape mask to the feature size
        mask_pos = F.interpolate(mask_train[0], size=(f_train.shape[-2], f_train.shape[-1]))
        mask_neg = 1 - mask_pos

        pred_pos, pred_neg = self.similarity_segmentation(f_test, f_train, mask_pos, mask_neg)

        pred_ = torch.cat((torch.unsqueeze(pred_pos, -1), torch.unsqueeze(pred_neg, -1)), dim=-1)
        pred_sm = F.softmax(pred_, dim=-1)

        if test_dist is not None:
            # distance map is give - resize for mixer
            dist = F.interpolate(test_dist[0], size=(f_train.shape[-2], f_train.shape[-1]))
            # concatenate inputs for mixer
            # softmaxed segmentation, positive segmentation and distance map
            segm_layers = torch.cat((torch.unsqueeze(pred_sm[:, :, :, 0], dim=1),
                                     torch.unsqueeze(pred_pos, dim=1),
                                     dist), dim=1)
        else:
            segm_layers = torch.cat((torch.unsqueeze(pred_sm[:, :, :, 0], dim=1), torch.unsqueeze(pred_pos, dim=1)), dim=1)

        out = self.mixer(segm_layers)
        out = self.s3(F.upsample(out, scale_factor=2))

        out = self.post2(F.upsample(self.f2(feat_test[2]) + self.s2(out), scale_factor=2))
        out = self.post1(F.upsample(self.f1(feat_test[1]) + self.s1(out), scale_factor=2))
        out = self.post0(F.upsample(self.f0(feat_test[0]) + self.s0(out), scale_factor=2))

        return out


    def similarity_segmentation(self, f_test, f_train, mask_pos, mask_neg):
        # first normalize train and test features to have L2 norm 1
        # cosine similarity and reshape last two dimensions into one
        sim = torch.einsum('ijkl,ijmn->iklmn',
                           F.normalize(f_test, p=2, dim=1),
                           F.normalize(f_train, p=2, dim=1))

        sim_resh = sim.view(sim.shape[0], sim.shape[1], sim.shape[2], sim.shape[3] * sim.shape[4])

        # reshape masks into vectors for broadcasting [B x 1 x 1 x w * h]
        # re-weight samples (take out positive ang negative samples)
        sim_pos = sim_resh * mask_pos.view(mask_pos.shape[0], 1, 1, -1)
        sim_neg = sim_resh * mask_neg.view(mask_neg.shape[0], 1, 1, -1)

        # take top k positive and negative examples
        # mean over the top positive and negative examples
        pos_map = torch.mean(torch.topk(sim_pos, self.topk_pos, dim=-1).values, dim=-1)
        neg_map = torch.mean(torch.topk(sim_neg, self.topk_neg, dim=-1).values, dim=-1)

        return pos_map, neg_map

