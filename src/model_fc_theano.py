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

lio = StringIO.StringIO()

def strnow():
    return datetime.datetime.now().strftime('%Y-%m-%d %H_%M_%S')

def safe_tofloat(s, default=.0):
    try:
        return float(s)
    except:
        return default

class Tone_Classification():
    def __init__(self, data, args, verbose=True):
        pprint(args, stream=lio)
        self.hidden_dim = args['hidden_dim']
        self.batch_size = args['batch_size']
        #self.input_dim = (args['num_wavepoint'] * 2 + args['num_slope']) 
        #self.input_dim = 13 + args['num_wavepoint'] * 2
        self.input_dim = args['num_wavepoint'] + args['mfcc']
        #self.input_dim = args['num_wavepoint']
        self.args = args
        self.out = sys.stdout if verbose else open(os.devnull, 'w')
        # data_process.strip_zeros(data, 0.05)
        #data_process.low_pass_filter(data)
        data_process.strip_zeros_by_energy(data, 0.15)
        #data_process.data_augmentation(data)
        if args['num_slope']:
            data_process.calc_segmented_slope(data, args['num_slope'])
        if args['num_wavepoint']:
            data_process.fix_length_by_interpolatation(data, args['num_wavepoint'], args['num_mfcc'])
        data_process.shuffle(data)
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
                    if not args['mfcc']:
                        concated = datum.f0 
                    else: 
                        concated = np.hstack((datum.f0, mfcc(np.array(datum.engy), args['rate'], numcep = args['mfcc'])[0]))
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

    def normlization(self, x):
        return x
        return list(np.array(x) * 10.0)
        return list(x / np.sqrt(np.sum(np.power(x, 2))))

    def get_params(self, layers):
        params = []
        for l in layers:
            if l.params != None:
                params.append(l.params)
        return params

    def set_params(self, layers, params):
        k = 0
        for l in layers:
            if l.params != None:
                l.params = params[k]
                k += 1

    def flatten(self, x):
        out = []
        for p in x:
            for y in p:
                out.append(y)
        return out

    def build_model(self, init_params=None):
        self.lr = T.scalar()
        if self.args['optimizer'] == 'sgd':
            opt = nnutils.SGDOptimizer(lr=self.lr)
        elif self.args['optimizer'] == 'adam':
            opt = nnutils.AdamOptimizer(lr=self.lr)
        else:
            raise RuntimeError('unsupported optimizer %s' % self.args['optimizer'])

        x = self.x = T.tensor4()
        y = self.y = T.ivector()
        self.layers = layers = []
        dim = self.input_dim
        layers.append(nnutils.Conv2D(
            input_shape=(self.batch_size, 1, dim, 1),
            filter_shape=(96, 1, 11, 1),
            act=nnutils.ReLU,
            zeropad=(5, 0)
        ))
        layers.append(nnutils.MaxPool2D(pool_shape=(2, 1)))
        dim = dim // 2
        layers.append(nnutils.Conv2D(
            input_shape=(self.batch_size, 1, dim, 1),
            filter_shape=(108, 96, 11, 1),
            act=nnutils.ReLU,
            zeropad=(5, 0)
        ))
        layers.append(nnutils.MaxPool2D(pool_shape=(2, 1)))
        dim = dim // 2
        layers.append(nnutils.Conv2D(
            input_shape=(self.batch_size, 1, dim, 1),
            filter_shape=(108, 108, 11, 1),
            act=nnutils.ReLU,
            zeropad=(5, 0)
        ))
        layers.append(nnutils.MaxPool2D(pool_shape=(2, 1)))
        dim = dim // 2
        layers.append(nnutils.Conv2D(
            input_shape=(self.batch_size, 1, dim, 1),
            filter_shape=(16, 108, 11, 1),
            act=nnutils.ReLU,
            zeropad=(5, 0)
        ))
        layers.append(nnutils.Flatten(2))
        dims = [dim * 16] + self.hidden_dim
        for i in xrange(len(self.hidden_dim)-1):
            layers.append(nnutils.FCLayer(dims[i], dims[i+1],
                                          act=nnutils.ReLU if i != len(self.hidden_dim)-2 else nnutils.linear))
            if i == 0 and self.args['dropout'] < 1:
                layers.append(nnutils.Dropout(self.args['dropout']))
        nnutils.Dropout.set_dropout_on(training=True)

        if init_params is not None:
            self.set_params(layers, init_params)
        L1_loss, L2_loss = 0., 0.
        h = x
        for layer in layers:
            if hasattr(layer, 'L2_Loss'):
                L2_loss += layer.L2_Loss
            elif hasattr(layer, 'L1_Loss'):
                L1_loss += layer.L1_Loss
            h = layer.forward(h)
        probs = self.probs = T.nnet.softmax(h)
        preds = self.preds = probs
        self.loss = -T.mean(T.log(probs[T.arange(y.shape[0]), y]))
        if self.args['use_L2']:
            self.loss += self.args['L2_reg'] * L2_loss
        elif self.args['use_L1']:
            self.loss += self.args['L1_reg'] * L1_loss

        self.lparams = self.get_params(self.layers)
        self.params = self.flatten(self.lparams)
        grads, updates = opt(self.loss, self.params)
        self.train_func = theano.function(
            inputs = [self.x, self.y, self.lr],
            outputs = [self.loss],
            updates = updates
        )
        self.eval_func = theano.function(
            inputs = [self.x, self.y],
            outputs = [self.preds, self.loss]
        )

    def get_batch(self, dataset, index):
        data_xs = self.data_xs[dataset]
        data_ys = self.data_ys[dataset]
        st = index * self.batch_size
        ed = st + self.batch_size
        if ed >= data_xs.shape[0]:
            return None, None
        ret_x = np.zeros((self.batch_size, self.input_dim), np.float32)
        ret_y = np.zeros((self.batch_size, ), np.int32)
        ret_x = data_xs[st:ed]
        ret_y = data_ys[st:ed]
        ret_x = ret_x.reshape(ret_x.shape[0], 1,  ret_x.shape[1], 1)
        return ret_x, ret_y

    def evaluate(self, dataset):
        batch_size = self.batch_size
        total_loss = 0.
        total_err = 0.
        n_batch = 0
        now_pos = 0
        nnutils.Dropout.set_dropout_on(training=False)
        while True:
            prepared_x, prepared_y = self.get_batch(dataset, n_batch)
            #print >> self.out, prepared_x, prepared_y
            if prepared_x is None:
                break
            preds, loss = self.eval_func(prepared_x, prepared_y)
            total_loss += np.mean(loss)
            for i in range(len(preds)):
                if np.argmax(preds[i]) != prepared_y[i]:
                    # print >> self.out, 'preds[i] =', preds[i], 'prepared_y[i] =', prepared_y[i]
                    total_err += 1
            n_batch += 1
        loss = total_loss / n_batch
        err = total_err / (n_batch * batch_size)
        print >> self.out, 'evaluate %s: loss = %f err = \033[1;31m%f\033[0m' % (dataset, loss, err)
        nnutils.Dropout.set_dropout_on(training=True)
        return loss, err

    def train(self, stop_if_hang=True):
        lr = self.args['init_lr']
        best_acc = 0

        val_loss_list = []
        for epoch in xrange(self.args['num_epoch']):
            n_train_batch = 0
            batch_size = self.args['batch_size']
            lr *= self.args['lr_decay']
            print >> self.out, '\033[1;36mThe epoch %d training: \033[0m' % epoch
            while True:
                prepared_x, prepared_y = self.get_batch('train', n_train_batch)
                if prepared_x is None:
                    break
                # print >> self.out, prepared_y
                loss, = self.train_func(prepared_x, prepared_y, lr)
                if n_train_batch % 100 == 0:
                    print >> self.out, 'The iteration is %d train loss is: %f' % (n_train_batch, loss)
                    # for param in self.params:
                    #     print >> self.out, param.get_value()
                if n_train_batch % 200 == 0:
                    loss, err = self.evaluate('test')
                    val_loss_list.append(loss)
                n_train_batch += 1
            loss, err = self.evaluate('test_new')
            acc = 1-err
            best_acc = max(best_acc, acc)

        with open('../doc/val_loss_theano.csv', 'w') as f:
            for epoch, loss in zip(xrange(self.args['num_epoch']), val_loss_list):
                f.write('%.16f,%.16f\n' % (epoch, loss))
        print 'best acc', best_acc

    def save(self, dirname):
        with open(os.path.join(dirname, 'model.bin'), 'w') as f:
            cPickle.dump(self.lparams, f, protocol=pickle.HIGHEST_PROTOCOL)


def run_single():
    global theano, T, nnutils
    import theano
    import theano.tensor as T
    import nnutils_theano as nnutils
    args = {
        'num_wavepoint': 100,
        'mfcc': 6,
        'rate': 70,
        'num_mfcc': 1000,
        'num_slope': 0,
        'lr_decay': 1.0,
        'dropout': 0.5,
        'hidden_dim': [128, 80, 4],
        'num_epoch': 200,
        'batch_size': 4,
        'optimizer': 'adam',
        'init_lr': 0.001,
        'use_L2': True,
        'use_L1': False,
        'L2_reg': 0.0005,
        'L1_reg': 0.0005,
    }
    data = data_process.read_all()
    model = Tone_Classification(data, args)
    model.build_model()
    acc = model.train()


if __name__ == "__main__":
    run_single()
