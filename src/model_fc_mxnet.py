from __future__ import division
import numpy as np
import copy
import itertools
import os
import sys
import time
import datetime
import multiprocessing
import data_process
import StringIO
from pprint import pprint
from python_speech_features import mfcc
#from python_speech_features import delta
from python_speech_features import logfbank

import logging
logging.basicConfig(level=logging.INFO)

class Tone_Classification():
    def __init__(self, data, args, verbose=True):
        self.hidden_dim = args['hidden_dim']
        self.batch_size = args['batch_size']
        #self.input_dim = (args['num_wavepoint'] * 2 + args['num_slope']) 
        #self.input_dim = 13 + args['num_wavepoint'] * 2
        self.input_dim = args['num_wavepoint']
        #self.input_dim = args['num_wavepoint']
        self.args = args
        self.out = sys.stdout if verbose else open(os.devnull, 'w')
        data_process.shuffle(data)
        # data_process.strip_zeros(data, 0.05)
        data_process.strip_zeros_by_energy(data, 0.15)
        if args['num_slope']:
            data_process.calc_segmented_slope(data, args['num_slope'])
        if args['num_wavepoint']:
            data_process.fix_length_by_interpolatation(data, args['num_wavepoint'])
        self.data_xs, self.data_ys = {}, {}
        for k, v in data.iteritems():
            xs, ys = [], []
            for datum in v:
                if args['num_slope']:
                    #engy = self.normlization(datum.engy)
                    #f0 = self.normlization(datum.f0)
                    engy = datum.engy
                    f0 = datum.f0
                    slope_engy = self.normlization(datum.slope_engy)
                    slope_f0 = self.normlization(datum.slope_f0)
                    concated = slope_engy 
                    #concated = engy + f0 + slope_f0
                else:
                    concated = datum.f0
                #assert len(concated) == self.input_dim
                xs.append(concated)
                #xs.append(concated) 
                ys.append(datum.tone)
                #dat = []
                #for a, b in zip(datum.engy, datum.f0):
                #    dat.append(a)
                #    dat.append(b)
                #xx1 = list(mfcc(np.array(dat))[0])
                #xx2 = list(logfbank(np.array(datum.f0))[0])
                #xs.append(datum.engy + datum.f0 + xx1)
                #ys.append(datum.tone)
            self.data_xs[k] = np.array(xs, np.float32)
            self.data_ys[k] = np.array(ys, np.int32) - 1
            print >> self.out, self.data_xs[k].shape, self.data_ys[k].shape

    def create_data_iter(self, dataset_name, batch_size='full'):
        xs = np.array(self.data_xs[dataset_name])
        xs = xs.reshape(xs.shape[0], 1, xs.shape[1], 1)
        xs = mx.nd.array(xs)
        # xs = mx.nd.array(self.data_xs[dataset_name])
        ys = mx.nd.array(self.data_ys[dataset_name])
        if batch_size == 'full':
            batch_size = len(self.data_xs[dataset_name])
        return mx.io.NDArrayIter(xs, label=ys, batch_size=batch_size)

    def build_model(self):
        if self.args.get('use_L2', False) or self.args.get('use_L1', False):
            raise RuntimeError('Regularization not supported')

        net = mx.sym.Variable('data')
        net = mx.sym.Convolution(net, kernel=(5, 1), num_filter=6)
        net = mx.sym.Activation(net, act_type='relu')
        net = mx.sym.Pooling(net, pool_type='max', kernel=(2, 1), stride=(2, 1))
        net = mx.sym.Convolution(net, kernel=(5, 1), num_filter=16)
        net = mx.sym.Activation(net, act_type='relu')
        net = mx.sym.Pooling(net, pool_type='max', kernel=(2, 1), stride=(2, 1))
        net = mx.sym.Flatten(net)
        for i, num_hidden in enumerate(self.hidden_dim):
            net = mx.sym.FullyConnected(net, num_hidden=num_hidden)
            if i != len(self.hidden_dim)-1:
                net = mx.sym.Activation(net, act_type='relu')
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
                        optimizer=self.args['optimizer'],
                        optimizer_params={'learning_rate': self.args['init_lr']},
                        eval_metric='acc',
                        num_epoch=self.args['num_epoch'])


def run_single():
    global mx
    import mxnet as mx
    args = {
        'num_wavepoint': 200,
        'num_slope': 0,
        'lr_decay': 0.99,
        'dropout': 1,
        'hidden_dim': [128, 42, 4],
        'num_epoch': 100,
        'batch_size': 4,
        'optimizer': 'adam',
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