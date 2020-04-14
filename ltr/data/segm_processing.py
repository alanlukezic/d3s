import torch
import torchvision.transforms as transforms
import numpy as np
from pytracking import TensorDict
import ltr.data.processing_utils as prutils
import random
import copy


class BaseProcessing:
    """ Base class for Processing. Processing class is used to process the data returned by a dataset, before passing it
     through the network. For example, it can be used to crop a search region around the object, apply various data
     augmentations, etc."""
    def __init__(self, transform=transforms.ToTensor(), train_transform=None, test_transform=None, joint_transform=None):
        """
        args:
            transform       - The set of transformations to be applied on the images. Used only if train_transform or
                                test_transform is None.
            train_transform - The set of transformations to be applied on the train images. If None, the 'transform'
                                argument is used instead.
            test_transform  - The set of transformations to be applied on the test images. If None, the 'transform'
                                argument is used instead.
            joint_transform - The set of transformations to be applied 'jointly' on the train and test images.  For
                                example, it can be used to convert both test and train images to grayscale.
        """
        self.transform = {'train': transform if train_transform is None else train_transform,
                          'test':  transform if test_transform is None else test_transform,
                          'joint': joint_transform}

    def __call__(self, data: TensorDict):
        raise NotImplementedError


class SegmProcessing(BaseProcessing):
    """ The processing class used for training ATOM. The images are processed in the following way.
    First, the target bounding box is jittered by adding some noise. Next, a square region (called search region )
    centered at the jittered target center, and of area search_area_factor^2 times the area of the jittered box is
    cropped from the image. The reason for jittering the target box is to avoid learning the bias that the target is
    always at the center of the search region. The search region is then resized to a fixed size given by the
    argument output_sz. A set of proposals are then generated for the test images by jittering the ground truth box.

    """
    def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor,
                 mode='pair', use_distance=False, *args, **kwargs):
        """
        args:
            search_area_factor - The size of the search region  relative to the target size.
            output_sz - An integer, denoting the size to which the search region is resized. The search region is always
                        square.
            center_jitter_factor - A dict containing the amount of jittering to be applied to the target center before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            scale_jitter_factor - A dict containing the amount of jittering to be applied to the target size before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            proposal_params - Arguments for the proposal generation process. See _generate_proposals for details.
            mode - Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames
        """
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.mode = mode
        self.use_distance = use_distance

    def _get_jittered_box(self, box, mode):
        """ Jitter the input box
        args:
            box - input bounding box
            mode - string 'train' or 'test' indicating train or test data

        returns:
            torch.Tensor - jittered box
        """

        jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        max_offset = (jittered_size.prod().sqrt() * self.center_jitter_factor[mode]).item()
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)

        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def _generate_proposals(self, box):
        """ Generates proposals by adding noise to the input box
        args:
            box - input box

        returns:
            torch.Tensor - Array of shape (num_proposals, 4) containing proposals
            torch.Tensor - Array of shape (num_proposals,) containing IoU overlap of each proposal with the input box. The
                        IoU is mapped to [-1, 1]
        """
        # Generate proposals
        num_proposals = self.proposal_params['boxes_per_frame']
        proposals = torch.zeros((num_proposals, 4))
        gt_iou = torch.zeros(num_proposals)

        for i in range(num_proposals):
            proposals[i, :], gt_iou[i] = prutils.perturb_box(box, min_iou=self.proposal_params['min_iou'],
                                                             sigma_factor=self.proposal_params['sigma_factor']
                                                             )

        # Map to [-1, 1]
        gt_iou = gt_iou * 2 - 1
        return proposals, gt_iou

    def _create_distance_map(self, map_sz, cx, cy, w, h, p=4, sz_weight=0.7):
        # create a square-shaped distance map with a Gaussian function which can be interpreted as a distance
        # to the given bounding box (center [cx, cy], width w, height h)
        # p is power of a Gaussian function
        # sz_weight is a weight of a bounding box size in Gaussian denominator
        x_ = np.linspace(1, map_sz, map_sz) - 1 - cx
        y_ = np.linspace(1, map_sz, map_sz) - 1 - cy
        X, Y = np.meshgrid(x_, y_)
        # 1 - is needed since we need distance-like map (not Gaussian function)
        return 1 - np.exp(-((np.power(X, p) / (sz_weight * w ** p)) + (np.power(Y, p) / (sz_weight * h ** p))))

    def _make_aabb_mask(self, map_shape, bbox):
        mask = np.zeros(map_shape, dtype=np.float32)
        mask[int(round(bbox[1].item())):int(round(bbox[1].item() + bbox[3].item())), int(round(bbox[0].item())):int(round(bbox[0].item() + bbox[2].item()))] = 1
        return mask

    def __call__(self, data: TensorDict):
        """
        args:
            data - The input data, should contain the following fields:
                'train_images'  -
                'test_images'   -
                'train_anno'    -
                'test_anno'     -
                'train_masks'   -
                'test_masks'    -

        returns:
            TensorDict - output data block with following fields:
                'train_images'  -
                'test_images'   -
                'train_anno'    -
                'test_anno'     -
                'train_masks'   -
                'test_masks'    -
        """
        # Apply joint transforms
        if self.transform['joint'] is not None:
            num_train_images = len(data['train_images'])
            all_images = data['train_images'] + data['test_images']
            all_images_trans = self.transform['joint'](*all_images)

            data['train_images'] = all_images_trans[:num_train_images]
            data['test_images'] = all_images_trans[num_train_images:]

        # extract patches from images
        for s in ['test', 'train']:#['train', 'test']:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num train/test frames must be 1"

            # Add a uniform noise to the center pos
            jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]

            # Crop image region centered at jittered_anno box
            crops_img, boxes = prutils.jittered_center_crop(data[s + '_images'], jittered_anno, data[s + '_anno'],
                                                self.search_area_factor, self.output_sz)

            # Crop mask region centered at jittered_anno box
            crops_mask, _ = prutils.jittered_center_crop(data[s + '_masks'], jittered_anno, data[s + '_anno'],
                                                            self.search_area_factor, self.output_sz, pad_val=float(0))

            if s == 'test' and self.use_distance:
                # use target center only to create distance map
                cx_ = (boxes[0][0] + boxes[0][2] / 2).item() + ((0.25 * boxes[0][2].item()) * (random.random() - 0.5))
                cy_ = (boxes[0][1] + boxes[0][3] / 2).item() + ((0.25 * boxes[0][3].item()) * (random.random() - 0.5))
                x_ = np.linspace(1, crops_img[0].shape[1], crops_img[0].shape[1]) - 1 - cx_
                y_ = np.linspace(1, crops_img[0].shape[0], crops_img[0].shape[0]) - 1 - cy_
                X, Y = np.meshgrid(x_, y_)
                D = np.sqrt(np.square(X) + np.square(Y)).astype(np.float32)

                data['test_dist'] = [torch.from_numpy(np.expand_dims(D, axis=0))]

            # Apply transforms
            data[s + '_images'] = [self.transform[s](x) for x in crops_img]
            data[s + '_anno'] = boxes
            data[s + '_masks'] = [torch.from_numpy(np.expand_dims(x, axis=0)) for x in crops_mask]

            if s == 'train' and random.random() < 0.005:
                # on random use binary mask generated from axis-aligned bbox
                data['test_images'] = copy.deepcopy(data['train_images'])
                data['test_masks'] = copy.deepcopy(data['train_masks'])
                data['test_anno'] = copy.deepcopy(data['train_anno'])
                data[s + '_masks'] = [torch.from_numpy(np.expand_dims(self._make_aabb_mask(x_.shape, bb_), axis=0)) for x_, bb_ in zip(crops_mask, boxes)]

                if self.use_distance:
                    # there is no need to randomly perturb center since we are working with ground-truth here
                    cx_ = (boxes[0][0] + boxes[0][2] / 2).item()
                    cy_ = (boxes[0][1] + boxes[0][3] / 2).item()
                    x_ = np.linspace(1, crops_img[0].shape[1], crops_img[0].shape[1]) - 1 - cx_
                    y_ = np.linspace(1, crops_img[0].shape[0], crops_img[0].shape[0]) - 1 - cy_
                    X, Y = np.meshgrid(x_, y_)
                    D = np.sqrt(np.square(X) + np.square(Y)).astype(np.float32)
                    data['test_dist'] = [torch.from_numpy(np.expand_dims(D, axis=0))]

        # Prepare output
        if self.mode == 'sequence':
            data = data.apply(prutils.stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

        return data