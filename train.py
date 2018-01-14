#!/usr/bin/env python3

import argparse
import logging

import matplotlib
matplotlib.use('Agg') # noqa / this should be before numpy/chainer

from chainer.ya.utils import rangelog, SourceBackup, ArgumentBackup, FinalRequest, SlackPost, SamplePairingDataset # noqa

import chainer
from chainer import functions as F
from chainer import links as L
from chainer import optimizers
from chainer import training
from chainer.datasets import get_cifar10
from chainer.training import StandardUpdater, extensions
from chainer.training.extensions import snapshot_object
from chainer.training.triggers import MinValueTrigger
from chainer.iterators.serial_iterator import SerialIterator


class Conv(chainer.Chain):
    def __init__(self, n_out, n_units=128, n_layers=10):
        super(Conv, self).__init__()
        with self.init_scope():
            for i in range(n_layers):
                setattr(self, "c"+str(i), L.Convolution2D(None,n_units, ksize=3, pad=1))
            self.l = L.Linear(None,n_out)
        self.n_units = n_units
        self.n_out = n_out
        self.n_layers = n_layers

    def predict(self, x):
        h = x
        for i in range(self.n_layers):
            h = F.relu(getattr(self,"c"+str(i))(h))
        return self.l(h)

    def __call__(self, x, t):
        loss = F.softmax_cross_entropy(self.predict(x), t)
        chainer.report({'loss': loss/t.shape[0]}, self)
        return loss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-S", "--sample_pairing", action="store_true", default=False)
    parser.add_argument("--model_path", default='model.npz')
    parser.add_argument("-e", "--epoch", type=int, default=10)
    parser.add_argument("-b", "--batch", type=int, default=500)
    parser.add_argument("--store", default="result")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--report_keys", action="append", default=['loss'])
    args = parser.parse_args()
    args.report_keys = ['main/'+k for k in args.report_keys]
    args.report_keys += ['validation/'+k for k in args.report_keys]
    return args


def train(args):
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, 'INFO'))
    logger.addHandler(logging.StreamHandler())
    rangelog.set_logger(logger)
    rangelog.set_start_msg("start... {name}")
    rangelog.set_end_msg("  end...")
    with rangelog("creating dataset") as logger:
        train_set, eval_set = get_cifar10()
        if args.sample_pairing:
            train_set = SamplePairingDataset(train_set)
    with rangelog("creating iterator") as logger:
        logger.info("train_set: {}, eval_set: {}"
                    .format(len(train_set), len(eval_set)))
        iterator = SerialIterator(train_set, args.batch, repeat=True)
        eval_iterator = SerialIterator(eval_set, args.batch, repeat=False)
    with rangelog("creating model") as logger:
        logger.info('GPU: {}'.format(args.device))
        model = Conv(10)
        chainer.cuda.get_device_from_id(args.device).use()
        model.to_gpu(args.device)
    with rangelog("creating optimizer"):
        optimizer = optimizers.Adam()
        optimizer.setup(model)
    with rangelog("creating trainer"):
        updater = StandardUpdater(iterator=iterator,
                                  optimizer=optimizer,
                                  device=args.device)
        trainer = training.Trainer(updater, (args.epoch, 'epoch'),
                                   out=args.store)
    with rangelog("trainer extension") as logger:
        trainer.extend(extensions.Evaluator(iterator=eval_iterator,
                                            target=model,
                                            device=args.device))
        trainer.extend(extensions.LogReport())
        trainer.extend(SourceBackup())
        trainer.extend(ArgumentBackup(args))
        trainer.extend(extensions.PrintReport(['epoch']+args.report_keys))
        trainer.extend(extensions.ProgressBar(update_interval=1))
        trainer.extend(extensions.PlotReport(args.report_keys, 'epoch',
                                             file_name='plot.png'))
        trigger = MinValueTrigger(key='validation/main/loss')
        snapshoter = snapshot_object(model, filename=args.model_path)
        trainer.extend(snapshoter, trigger=trigger)
    with rangelog("training"):
        trainer.run()
    return model


if __name__ == '__main__':
    train(parse_args())
