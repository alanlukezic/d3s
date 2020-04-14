import torch.nn as nn
import ltr.models.backbone as backbones
import ltr.models.segm as segmmodels
from ltr import model_constructor


class SegmNet(nn.Module):
    """ Segmentation network module"""
    def __init__(self, feature_extractor, segm_predictor, segm_layers, extractor_grad=True):
        """
        args:
            feature_extractor - backbone feature extractor
            segm_predictor - segmentation module
            segm_layers - List containing the name of the layers from feature_extractor, which are used in segm_predictor
            extractor_grad - Bool indicating whether backbone feature extractor requires gradients
        """
        super(SegmNet, self).__init__()

        self.feature_extractor = feature_extractor
        self.segm_predictor = segm_predictor
        self.segm_layers = segm_layers

        if not extractor_grad:
            for p in self.feature_extractor.parameters():
                p.requires_grad_(False)

    def forward(self, train_imgs, test_imgs, train_masks, test_dist=None):
        """ Forward pass
        Note: If the training is done in sequence mode, that is, test_imgs.dim() == 5, then the batch dimension
        corresponds to the first dimensions. test_imgs is thus of the form [sequence, batch, feature, row, col]
        """

        train_feat = self.extract_backbone_features(train_imgs)
        test_feat = self.extract_backbone_features(test_imgs)

        train_feat_segm = [feat for feat in train_feat.values()]
        test_feat_segm = [feat for feat in test_feat.values()]
        train_masks = [train_masks]

        if test_dist is not None:
            test_dist = [test_dist]

        # Obtain iou prediction
        segm_pred = self.segm_predictor(test_feat_segm, train_feat_segm, train_masks, test_dist)
        return segm_pred

    def extract_backbone_features(self, im, layers=None):
        if layers is None:
            layers = self.segm_layers
        return self.feature_extractor(im, layers)

    def extract_features(self, im, layers):
        return self.feature_extractor(im, layers)



@model_constructor
def segm_resnet18(segm_input_dim=(256,256), segm_inter_dim=(256,256), backbone_pretrained=True, topk_pos=3, topk_neg=3, mixer_channels=2):
    # backbone
    backbone_net = backbones.resnet18(pretrained=backbone_pretrained)

    # segmentation dimensions
    segm_input_dim = (64, 64, 128, 256)
    segm_inter_dim = (4, 16, 32, 64)
    segm_dim = (64, 64)  # convolutions before cosine similarity

    # segmentation
    segm_predictor = segmmodels.SegmNet(segm_input_dim=segm_input_dim,
                                            segm_inter_dim=segm_inter_dim,
                                            segm_dim=segm_dim,
                                            topk_pos=topk_pos,
                                            topk_neg=topk_neg,
                                            mixer_channels=mixer_channels)

    net = SegmNet(feature_extractor=backbone_net, segm_predictor=segm_predictor,
                      segm_layers=['conv1', 'layer1', 'layer2', 'layer3'], extractor_grad=False)

    return net


@model_constructor
def segm_resnet50(segm_input_dim=(256,256), segm_inter_dim=(256,256), backbone_pretrained=True, topk_pos=3, topk_neg=3, mixer_channels=2):
    # backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained)

    # segmentation dimensions
    segm_input_dim = (64, 256, 512, 1024)
    segm_inter_dim = (4, 16, 32, 64)
    segm_dim = (64, 64)  # convolutions before cosine similarity

    # segmentation
    segm_predictor = segmmodels.SegmNet(segm_input_dim=segm_input_dim,
                                            segm_inter_dim=segm_inter_dim,
                                            segm_dim=segm_dim,
                                            topk_pos=topk_pos,
                                            topk_neg=topk_neg,
                                            mixer_channels=mixer_channels)

    net = SegmNet(feature_extractor=backbone_net, segm_predictor=segm_predictor,
                      segm_layers=['conv1', 'layer1', 'layer2', 'layer3'], extractor_grad=False)  # extractor_grad=False

    return net
