import numpy as np
import copy
import itertools
import os
import sys
import time
import multiprocessing
import data_process
from pprint import pprint
import logging
logging.basicConfig(level=logging.INFO)

class Tone_Classification():
    def __init__(self, data, args, verbose=True):
        self.hidden_dim = args['hidden_dim']
        self.batch_size = args['batch_size']
        self.input_dim = (args['num_wavepoint'] + args['num_slope']) * 2
        self.args = args
        self.out = sys.stdout if verbose else open(os.devnull, 'w')
        data_process.shuffle(data)
        data_process.strip_zeros(data, 0.05)
        data_process.calc_segmented_slope(data, args['num_slope'])
        data_process.fix_length(data, args['num_wavepoint'], np.max)
        self.data_xs, self.data_ys = {}, {}
        for k, v in data.iteritems():
            xs, ys = [], []
            for datum in v:
                concated = datum.engy + datum.f0 + datum.slope_engy + datum.slope_f0
                assert len(concated) == self.input_dim
                xs.append(concated)
                ys.append(datum.tone)
            self.data_xs[k] = np.array(xs, np.float32)
            self.data_ys[k] = np.array(ys, np.int32) - 1
            print >> self.out, self.data_xs[k].shape, self.data_ys[k].shape

    def create_data_iter(self, dataset_name, batch_size='full'):
        xs = mx.nd.array(self.data_xs[dataset_name])
        ys = mx.nd.array(self.data_ys[dataset_name])
        if batch_size == 'full':
            batch_size = len(self.data_xs[dataset_name])
        return mx.io.NDArrayIter(xs, label=ys, batch_size=batch_size)

    def build_model(self):
        if self.args.get('use_L2', False) or self.args.get('use_L1', False):
            raise RuntimeError('Regularization not supported')

        net = mx.sym.Variable('data')
        for i, num_hidden in enumerate(self.hidden_dim):
            net = mx.sym.FullyConnected(net, name='fc%d' % (i+1), num_hidden=num_hidden)
            net = mx.sym.Activation(net, name='relu%d' % (i+1), act_type='relu')
        net = mx.sym.SoftmaxOutput(net, name='softmax')
        self.module = mx.mod.Module(symbol=net,
                                    context=mx.cpu(),
                                    data_names=['data'],
                                    label_names=['softmax_label'])

    def evaluate(self, dataset):
        data_iter = self.create_data_iter('test_new', batch_size=self.args['batch_size'])
        total_err, total_label = 0, 0
        for preds, batch_i, batch in self.module.iter_predict(data_iter):
            pred_label = preds[0].asnumpy().argmax(axis=1)
            label = batch.label[0].asnumpy().astype('int32')
            total_err += sum(pred_label != label)
            total_label += len(label)
        err = float(total_err) / len(label)
        print >> self.out, 'evaluate %s: err = \033[1;31m%f\033[0m' % (dataset, err)
        return err

    def train(self):
        data_train_iter = self.create_data_iter('train', batch_size=self.args['batch_size'])
        data_val_iter = self.create_data_iter('test', batch_size=self.args['batch_size'])
        self.module.fit(data_train_iter, eval_data=data_val_iter,
                        optimizer='sgd',
                        optimizer_params={'learning_rate': self.args['init_lr']},
                        eval_metric='acc',
                        num_epoch=self.args['num_epoch'])


def run_single():
    global mx
    import mxnet as mx
    args = {
        'num_wavepoint': 6,
        'num_slope': 2,
        'hidden_dim': [30, 15, 4],
        'num_epoch': 20,
        'batch_size': 2,
        'optimizer': 'sgd',
        'init_lr': 0.001,
        'use_L2': False,
        'use_L1': False,
        'L2_reg': 0.005,
        'L1_reg': 0.0005,
    }
    data = data_process.read_all()
    model = Tone_Classification(data, args)
    model.build_model()
    model.train()
    model.evaluate('test_new')


if __name__ == "__main__":
    run_single()
