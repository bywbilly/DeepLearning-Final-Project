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
        self.input_dim = args['num_wavepoint']
        self.args = args
        self.out = sys.stdout if verbose else open(os.devnull, 'w')
        data_process.strip_zeros_by_energy(data, 0.15)
        data_process.fix_length(data, args['num_wavepoint'], np.max)
        data_process.shuffle(data)
        self.data_xs, self.data_ys = {}, {}
        for k, v in data.iteritems():
            xs, ys = [], []
            for datum in v:
                xs.append(datum.f0)
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
        if self.args.get('use_L1', False):
            raise RuntimeError('Regularization not supported')

        net = mx.sym.Variable('data')
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
        data_iter = self.create_data_iter(dataset, batch_size=self.args['batch_size'])
        eval_metric = mx.metric.CompositeEvalMetric()
        eval_metric.add(mx.metric.Accuracy())
        eval_metric.add(mx.metric.CrossEntropy())
        score = self.module.score(data_iter, eval_metric)
        score = { k: v for k, v in score }
        err = 1 - score['accuracy']
        loss = score['cross-entropy']
        print >> self.out, 'evaluate %s: loss = %f err = \033[1;31m%f\033[0m' % (dataset, loss, err)
        return loss, err

    def train(self):
        self.val_loss_list = []
        self.best_acc = 0
        data_train_iter = self.create_data_iter('train', batch_size=self.args['batch_size'])
        data_val_iter = self.create_data_iter('test', batch_size=self.args['batch_size'])
        self.module.fit(data_train_iter, eval_data=data_val_iter,
                        optimizer=self.args['optimizer'],
                        optimizer_params={
                            'learning_rate': self.args['init_lr'],
                        },
                        eval_metric='acc',
                        num_epoch=self.args['num_epoch'],
                        epoch_end_callback=self.train_epoch_end_callback)

        with open('../doc/val_loss_mxnet.csv', 'w') as f:
            for epoch, loss in zip(xrange(self.args['num_epoch']), self.val_loss_list):
                f.write('%.16f,%.16f\n' % (epoch, loss))
        print 'best acc', self.best_acc

    def train_epoch_end_callback(self, epoch, symbol, arg_params, aux_states):
        loss_val, err_val = self.evaluate('test')
        loss_test, err_test = self.evaluate('test_new')
        acc_test = 1-err_test
        self.best_acc = max(self.best_acc, acc_test)
        self.val_loss_list.append(loss_val)


def run_single():
    global mx
    import mxnet as mx
    args = {
        'num_wavepoint': 12,
        'lr_decay': 1.0,
        'hidden_dim': [40, 15, 4],
        'num_epoch': 200,
        'batch_size': 4,
        'optimizer': 'sgd',
        'init_lr': 0.0004,
        'use_L2': False,
        'use_L1': False,
        'L2_reg': 0.0005,
        'L1_reg': 0.0005,
    }
    data = data_process.read_all()
    model = Tone_Classification(data, args)
    t1 = time.time()
    model.build_model()
    t2 = time.time()
    model.train()
    t3 = time.time()
    model.evaluate('test_new')
    print 'build_model time %.16fs' % (t2 - t1)
    print 'train %s epoch time %.16fs' % (args['num_epoch'], t3 - t2)


if __name__ == "__main__":
    run_single()