import numpy as np
from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList
import pandas


def VOT16Dataset():
    return VOTDatasetClass().get_sequence_list()


class VOTDatasetClass(BaseDataset):
    """VOT2016 dataset
    """
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.vot16_path
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        sequence_path = sequence_name
        nz = 8
        ext = 'jpg'
        start_frame = 1

        anno_path = '{}/{}/groundtruth.txt'.format(self.base_path, sequence_name)
        try:
            ground_truth_rect = np.loadtxt(str(anno_path), dtype=np.float64)
        except:
            ground_truth_rect = np.loadtxt(str(anno_path), delimiter=',', dtype=np.float64)

        end_frame = ground_truth_rect.shape[0]

        frames = ['{base_path}/{sequence_path}/{frame:0{nz}}.{ext}'.format(base_path=self.base_path,
                  sequence_path=sequence_path, frame=frame_num, nz=nz, ext=ext)
                  for frame_num in range(start_frame, end_frame+1)]

        return Sequence(sequence_name, frames, ground_truth_rect)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        list_path = '{}/list.txt'.format(self.base_path)
        sequence_list = pandas.read_csv(list_path, header=None, squeeze=True).values.tolist()

        return sequence_list
