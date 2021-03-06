import numpy as np
import os

from chainercv.chainer_experimental.datasets.sliceable import GetterDataset
from chainercv.utils import read_image
import pandas as pd

from chainer_epic_kitchens.datasets import epic_kitchens_utils


class EpicKitchensBboxDataset(GetterDataset):

    def __init__(
            self, data_dir='auto', year='2018', split='train',
            use_empty=False
    ):
        super(EpicKitchensBboxDataset, self).__init__()
        if data_dir == 'auto':
            data_dir = epic_kitchens_utils.get_epic_kitchens(year)

        if split == 'train':
            anno_file = os.path.join(
                data_dir, 'annotations', 'EPIC_train_object_labels.csv')
            self.anno_df = pd.read_csv(anno_file)
            self.img_dir = os.path.join(
                data_dir, 'object_detection_images/train')
        elif split == 'val':
            self.anno_df = None
            self.img_dir = os.path.join(
                data_dir, 'object_detection_images/test')

        self.ids = []
        for part_d in sorted(os.listdir(self.img_dir)):
            part_dir = os.path.join(self.img_dir, part_d)
            if not os.path.isdir(part_dir):
                continue
            for video_d in sorted(os.listdir(part_dir)):
                video_dir = os.path.join(part_dir, video_d)
                if not os.path.isdir(video_dir):
                    continue
                for filename in sorted(os.listdir(video_dir)):
                    filepath = os.path.join(video_dir, filename)
                    if not filename.endswith('.jpg'):
                        continue
                    if not os.path.isfile(filepath):
                        continue
                    self.ids.append(
                        '{0}/{1}/{2}'.format(part_d, video_d,
                                             filename.split('.')[0]))

        if self.anno_df is not None and not use_empty:
            anno_ids = self.anno_df[self.anno_df['bounding_boxes'] != '[]']
            anno_ids = anno_ids[['participant_id', 'video_id', 'frame']].values
            anno_ids[:, 1] = [x[4:] for x in anno_ids[:, 1]]
            anno_ids[:, 2] = ['{0:010d}'.format(x) for x in anno_ids[:, 2]]
            anno_ids = ['/'.join(x) for x in anno_ids]
            self.ids = sorted(list(set(self.ids) & set(anno_ids)))

        self.add_getter('img', self._get_image)
        self.add_getter(('bbox', 'label'), self._get_annotations)

    def __len__(self):
        return len(self.ids)

    def _get_image(self, i):
        img_file = os.path.join(
            self.img_dir, '{}.jpg'.format(self.ids[i]))
        return read_image(img_file, color=True)

    def _get_annotations(self, i):
        if self.anno_df is None:
            bbox = np.empty((0, 4), dtype=np.float32)
            label = np.empty((0, ), dtype=np.int32)
            return bbox, label
        part_id, video_id, frame_id = self.ids[i].split('/')
        video_id = '{0}_{1}'.format(part_id, video_id)
        anno_mask = \
            (self.anno_df['participant_id'] == part_id) \
            & (self.anno_df['video_id'] == video_id)  \
            & (self.anno_df['frame'] == int(frame_id))
        anno_data = self.anno_df[anno_mask]
        bbox = []
        label = []
        for bb_str, lbl in anno_data[['bounding_boxes', 'noun_class']].values:
            if lbl == 0:
                continue
            bb = eval(bb_str)
            if len(bb) == 0:
                continue
            for bb_ in bb:
                bbox.append(list(bb_))
                label.append(lbl - 1)
        if len(bbox) == 0:
            bbox = np.empty((0, 4), dtype=np.float32)
        else:
            bbox = np.array(bbox, dtype=np.float32)
            bbox[:, 2:4] += bbox[:, 0:2]
        label = np.array(label, dtype=np.int32)
        return bbox, label
