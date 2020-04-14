import os
import os.path
import torch
import numpy as np
import pandas
import csv
import glob
from PIL import Image
from collections import OrderedDict
from .base_dataset import BaseDataset
from ltr.data.image_loader import default_image_loader
from ltr.admin.environment import env_settings


class Vos(BaseDataset):
    """ VOS dataset (Video Object Segmentation).
    """

    def __init__(self, root=None, image_loader=default_image_loader, vid_ids=None, split=None):
        """
        args:
            root - path to the VOS dataset.
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            vid_ids - List containing the ids of the videos (1 - 20) used for training. If vid_ids = [1, 3, 5], then the
                    videos with subscripts -1, -3, and -5 from each class will be used for training.
            split - If split='train', the official train split (protocol-II) is used for training. Note: Only one of
                    vid_ids or split option can be used at a time.
        """
        root = env_settings().vos_dir if root is None else root
        super().__init__(root, image_loader)

        self.sequence_list = self._build_sequence_list(vid_ids, split)
        self.frame_names_dict, self.mask_names_dict = self._build_frames_list()


    def _build_sequence_list(self, vid_ids=None, split=None):
        if split != 'train' and split != 'val':
            print('Error: unknown VOS dataset split.')
            exit(-1)

        file_path = os.path.join(self.root, 'vos-list-' + split + '.txt')

        sequence_list = pandas.read_csv(file_path, header=None, squeeze=True).values.tolist()
        # each entry (sequence name) looks like this: folder_name-object_id
        # this should be parsed later, when sequence is loading

        return sequence_list


    def _build_frames_list(self):
        frame_names_dict = {}
        mask_names_dict = {}
        for seq_name in self.sequence_list:
            dir_name = seq_name.split('-')[0]
            # sequence frames path
            frames_path = os.path.join(self.root, 'JPEGImages', dir_name)
            # sequence masks path
            masks_path = os.path.join(self.root, 'Annotations', dir_name)

            frame_names_dict[seq_name] = sorted([file_name for file_name in glob.glob(os.path.join(frames_path, '*.jpg' ))])
            mask_names_dict[seq_name] = sorted([file_name for file_name in glob.glob(os.path.join(masks_path, '*.png'))])
        return frame_names_dict, mask_names_dict


    def get_name(self):
        return 'VOS'


    def get_num_sequences(self):
        return len(self.sequence_list)


    def _read_anno(self, seq_path, obj_id):
        anno_file = os.path.join(seq_path, "groundtruth-%s.txt" % obj_id)
        gt = pandas.read_csv(anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        return torch.tensor(gt)


    def _read_target_visible(self, seq_path, anno):
        return (anno[:, 0] > -1) & (anno[:, 1] > -1) & (anno[:, 2] > -1) & (anno[:, 3] > -1)


    def get_sequence_info(self, seq_id):

        seq_name = self.sequence_list[seq_id]
        dir_name = seq_name.split('-')[0]
        object_id = seq_name.split('-')[1]

        seq_path = os.path.join(self.root, 'Annotations', dir_name)
        anno = self._read_anno(seq_path, object_id)
        target_visible = self._read_target_visible(seq_path, anno)

        return anno, target_visible


    def _get_frame(self, frames_path, frame_id):
        return self.image_loader(frames_path[frame_id])


    def _get_mask(self, masks_path, frame_id, obj_id):
        m_ = np.asarray(Image.open(masks_path[frame_id])).astype(np.float32)
        mask = (m_ == float(obj_id)).astype(np.float32)
        return np.expand_dims(mask, axis=-1)


    def get_frames(self, seq_id, frame_ids, anno=None):
        # sequence dir name and object id
        seq_name = self.sequence_list[seq_id]
        dir_name = seq_name.split('-')[0]
        object_id = seq_name.split('-')[1]

        frames_path = self.frame_names_dict[seq_name]
        masks_path = self.mask_names_dict[seq_name]

        frame_list = [self._get_frame(frames_path, f_id) for f_id in frame_ids]
        mask_list = [self._get_mask(masks_path, f_id, object_id) for f_id in frame_ids]

        if anno is None:
            anno = self._read_anno(masks_path, object_id)

        # Return as list of tensors
        anno_frames = [anno[f_id, :] for f_id in frame_ids]

        object_meta = OrderedDict({'object_class': None,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        return frame_list, mask_list, anno_frames, object_meta
