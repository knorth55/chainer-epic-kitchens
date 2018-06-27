import argparse
import numpy as np

from chainer.backends import cuda
from chainercv.links import SSD300
from chainercv.links import SSD512
from chainercv.visualizations import vis_bbox
import matplotlib.pyplot as plt

from chainer_epic_kitchens.datasets import epic_kitchens_bbox_label_names
from chainer_epic_kitchens.datasets import EpicKitchensBboxDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', choices=('ssd300', 'ssd512'), default='ssd300')
    parser.add_argument(
        '--pretrained-model', default='imagenet')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--random', action='store_true')
    parser.add_argument('--split', choices=('train', 'val'), default='val')
    parser.add_argument('--skip', action='store_true')
    parser.add_argument('--score-thresh', type=float, default=0.6)
    args = parser.parse_args()

    if args.model == 'ssd300':
        model = SSD300(
            n_fg_class=len(epic_kitchens_bbox_label_names),
            pretrained_model=args.pretrained_model)
    elif args.model == 'ssd512':
        model = SSD512(
            n_fg_class=len(epic_kitchens_bbox_label_names),
            pretrained_model=args.pretrained_model)
    model.score_thresh = args.score_thresh
    if args.gpu > 0:
        cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    dataset = EpicKitchensBboxDataset(split=args.split)
    indices = np.arange(len(dataset))
    if args.random:
        np.random.shuffle(indices)
    for i in indices:
        img, _, _ = dataset[i]
        bboxes, labels, scores = model.predict([img])
        bbox, label, score = bboxes[0], labels[0], scores[0]
        if args.skip:
            if len(bbox) == 0:
                print('skip {}.jpg'.format(dataset.ids[i]))
                continue
        vis_bbox(
            img, bbox, label, score,
            label_names=epic_kitchens_bbox_label_names)
        plt.show()


if __name__ == '__main__':
    main()
