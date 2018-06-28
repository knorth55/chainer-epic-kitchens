import argparse
import os

import chainer
from chainer.backends import cuda
from chainercv.links import SSD300
from chainercv.links import SSD512
from chainercv.visualizations import vis_bbox
import matplotlib.pyplot as plt

from chainer_epic_kitchens.datasets import epic_kitchens_bbox_label_names
from chainer_epic_kitchens.datasets import EpicKitchensBboxDataset


thisdir = os.path.abspath(os.path.dirname(__file__))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', choices=('ssd300', 'ssd512'), default='ssd300')
    parser.add_argument(
        '--pretrained-model', default='imagenet')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--no-display', action='store_true')
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--save-path', type=str, default=None)
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
    if args.gpu >= 0:
        cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    dataset = EpicKitchensBboxDataset(split=args.split)
    dataset_iter = chainer.iterators.SerialIterator(
        dataset, args.batch_size, shuffle=args.shuffle, repeat=False)

    dataset_iter.reset()
    if dataset_iter._order is None:
        indices = dataset.ids
    else:
        indices = dataset_iter._order

    for batch_data in dataset_iter:
        imgs = []
        for data in batch_data:
            img, _, _ = data
            imgs.append(img)
        bboxes, labels, scores = model.predict(imgs)

        base_index = dataset_iter.current_position - args.batch_size
        for b_i in range(args.batch_size):
            img = imgs[b_i]
            bbox, label, score = bboxes[b_i], labels[b_i], scores[b_i]
            if args.skip:
                if len(bbox) == 0:
                    print('skip {}.jpg'.format(indices[base_index + b_i]))
                    continue
            vis_bbox(
                img, bbox, label, score,
                label_names=epic_kitchens_bbox_label_names)

            if args.save_path is not None:
                save_path = os.path.join(thisdir, args.save_path)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                plt.savefig(
                    os.path.join(
                        save_path,
                        'vis_{}.png'.format(
                            dataset.ids[base_index + b_i].replace('/', '_'))))
            if not args.no_display:
                plt.show()


if __name__ == '__main__':
    main()
