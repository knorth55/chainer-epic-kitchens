import numpy as np
import os

from chainercv.chainer_experimental.datasets.sliceable import GetterDataset
from chainercv.utils import read_image
import pandas as pd


import epic_kitchens_utils


class EpicKitchensBboxDataset(GetterDataset):

    def __init__(self, data_dir='auto', year='2018', split='train'):
        super(EpicKitchensBboxDataset, self).__init__()
        if data_dir == 'auto':
            data_dir = epic_kitchens_utils.get_epic_kitchens(year)

        if split == 'train':
            anno_file = os.path.join(
                data_dir, 'annotations', 'EPIC_train_object_labels.csv')
            self.anno_df = pd.read_csv(anno_file)
            self.img_dir = os.path.join(
                data_dir, 'object_detection_images/train')
        else:
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

        self.add_getter('img', self._get_image)
        self.add_getter(('bbox', 'label'), self._get_annotations)

    def __len__(self):
        return len(self.ids)

    def _get_image(self, i):
        img_file = os.path.join(
            self.img_dir, '{}.jpg'.format(self.ids[i]))
        return read_image(img_file, color=True)

    def _get_annotations(self, i):
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
            bb = eval(bb_str)
            if len(bb) == 0:
                continue
            for bb_ in bb:
                bbox.append(list(bb_))
                label.append(lbl)
        bbox = np.array(bbox, dtype=np.float32)
        if len(bbox) > 0:
            bbox = bbox[:, [0, 1, 3, 2]]
            bbox[:, 2:4] += bbox[:, 0:2]
        label = np.array(label, dtype=np.int32)
        return bbox, label
