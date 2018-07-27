import argparse
import numpy as np

import chainer
from chainer import serializers
from chainer import training
from chainer.training import extensions
from chainer.training import triggers
from chainercv.chainer_experimental.datasets.sliceable import TransformDataset
from chainercv.links import FasterRCNNVGG16
from chainercv.links.model.faster_rcnn import FasterRCNNTrainChain
from chainercv import transforms
import chainermn

from chainer_epic_kitchens.datasets import epic_kitchens_bbox_label_names
from chainer_epic_kitchens.datasets import EpicKitchensBboxDataset


class Transform(object):

    def __init__(self, faster_rcnn):
        self.faster_rcnn = faster_rcnn

    def __call__(self, in_data):
        img, bbox, label = in_data
        _, H, W = img.shape
        img = self.faster_rcnn.prepare(img)
        _, o_H, o_W = img.shape
        scale = o_H / H
        bbox = transforms.resize_bbox(bbox, (H, W), (o_H, o_W))

        # horizontally flip
        img, params = transforms.random_flip(
            img, x_random=True, return_param=True)
        bbox = transforms.flip_bbox(
            bbox, (o_H, o_W), x_flip=params['x_flip'])

        return img, bbox, label, scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--out', default='result')
    parser.add_argument('--resume')
    args = parser.parse_args()

    comm = chainermn.create_communicator()
    device = comm.intra_rank

    faster_rcnn = FasterRCNNVGG16(
        n_fg_class=len(epic_kitchens_bbox_label_names),
        pretrained_model='imagenet')

    faster_rcnn.use_preset('evaluate')
    model = FasterRCNNTrainChain(faster_rcnn)
    chainer.cuda.get_device_from_id(device).use()
    model.to_gpu()

    train = EpicKitchensBboxDataset(year='2018', split='train')
    if comm.rank == 0:
        indices = np.arange(len(train))
    else:
        indices = None
    train = TransformDataset(
        train, ('img', 'bbox', 'label', 'scale'),
        Transform(faster_rcnn))

    indices = chainermn.scatter_dataset(indices, comm, shuffle=True)
    train = train.slice[indices]

    train_iter = chainer.iterators.SerialIterator(
        train, batch_size=args.batchsize)

    optimizer = chainermn.create_multi_node_optimizer(
        chainer.optimizers.MomentumSGD(), comm)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(rate=0.0005))

    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, device=device)
    trainer = training.Trainer(updater, (18, 'epoch'), args.out)
    trainer.extend(
        extensions.ExponentialShift('lr', 0.1, init=args.lr),
        trigger=triggers.ManualScheduleTrigger([12, 15], 'epoch'))

    if comm.rank == 0:
        log_interval = 10, 'iteration'
        trainer.extend(extensions.LogReport(
            log_name='log.json', trigger=log_interval))
        trainer.extend(extensions.observe_lr(), trigger=log_interval)
        trainer.extend(extensions.PrintReport(
            ['iteration', 'epoch', 'elapsed_time', 'lr',
             'main/loss',
             'main/roi_loc_loss',
             'main/roi_cls_loss',
             'main/rpn_loc_loss',
             'main/rpn_cls_loss']), trigger=log_interval)
        trainer.extend(extensions.ProgressBar(update_interval=1))

        trainer.extend(
            extensions.snapshot_object(
                model.faster_rcnn, 'model_iter_{.updater.iteration}.npz'),
            trigger=(1, 'epoch'))

    if args.resume:
        serializers.load_npz(args.resume, trainer)

    trainer.run()


if __name__ == '__main__':
    main()
