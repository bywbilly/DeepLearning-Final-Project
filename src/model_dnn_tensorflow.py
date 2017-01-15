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

    def _loss(self, logits, L1_loss, L2_loss, labels):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits, labels, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        if self.args['use_L2']:
            cross_entropy_mean += self.args['L2_reg'] * L2_loss
        elif self.args['use_L1']:
            cross_entropy_mean += self.args['L1_reg'] * L1_loss
        return cross_entropy_mean

    def _forward(self, batch_x):
        layers = []
        dims = self.hidden_dim
        layers.append(tfnnutils.InputLayer())
        for i in xrange(len(self.hidden_dim)):
            layers.append(tfnnutils.FCLayer('FC%d'%(i+1), dims[i],
                                            act=tf.nn.relu if i != len(self.hidden_dim)-1 else None))



        L1_loss, L2_loss = 0., 0.
        last_layer = None
        for i, layer in enumerate(layers):
            if hasattr(layer, 'L2_Loss'):
                L2_loss += layer.L2_Loss
            elif hasattr(layer, 'L1_Loss'):
                L1_loss += layer.L1_Loss
            batch_x = layer.forward(last_layer, batch_x)
            last_layer = layer
            print >> self.out, layer
            print >> lio, layer
            #print >> self.out, batch_x
        
        pred = tf.nn.softmax(batch_x)
        #print >> self.out, pred
        
        return pred, batch_x, L1_loss, L2_loss
    
    def build_model(self):
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)
        self.lr = tf.placeholder(tf.float32, shape=[])
        if self.args['optimizer'] == 'sgd':
            opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        elif self.args['optimizer'] == 'momentum':
            opt = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=0.9)
        elif self.args['optimizer'] == 'adam':
            opt = tf.train.AdamOptimizer(learning_rate=self.lr)
        elif self.args['optimizer'] == 'adagrad':
            opt = tf.train.AdagradOptimizer(learning_rate=self.lr)
        else:
            raise RuntimeError('unsupported optimizer %s' % self.args['optimizer'])
        self._x = tf.placeholder(tf.float32, shape=[self.batch_size, self.input_dim])
        self._y = tf.placeholder(tf.int32)
        x = self._x
        y = self._y

        pred, logits, L1_loss, L2_loss = self._forward(x)
        loss = self._loss(logits, L1_loss, L2_loss, y)

        grads = opt.compute_gradients(loss)

        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        #init = tf.global_variables_initializer()
        init = tf.initialize_all_variables() 

        config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=False)
        config.gpu_options.allow_growth=True
        self.sess = tf.Session(config=config)

        self.sess.run(init)
        tf.train.start_queue_runners(sess=self.sess)

        self.train_step = apply_gradient_op
        self.pred_step = pred
        self.loss_step = loss

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
        return ret_x, ret_y

    def evaluate(self, dataset):
        batch_size = self.batch_size
        total_loss = 0.
        total_err = 0.
        n_batch = 0
        now_pos = 0
        while True:
            prepared_x, prepared_y = self.get_batch(dataset, n_batch)
            #print >> self.out, prepared_x, prepared_y
            if prepared_x is None:
                break
            feed = {self._x: prepared_x, self._y: prepared_y}
            loss, preds = self.sess.run([self.loss_step, self.pred_step], feed_dict=feed)
            total_loss += np.mean(loss)
            for i in range(len(preds)):
                if np.argmax(preds[i]) != prepared_y[i]:
                    # print >> self.out, 'preds[i] =', preds[i], 'prepared_y[i] =', prepared_y[i]
                    total_err += 1
            n_batch += 1
        loss = total_loss / n_batch
        err = total_err / (n_batch * batch_size)
        print >> self.out, 'evaluate %s: loss = %f err = \033[1;31m%f\033[0m' % (dataset, loss, err)
        print >> lio, 'evaluate %s: loss = %f err = %f' % (dataset, loss, err)
        return loss, err

    def train(self):
        global persistent_best_acc
        lr = self.args['init_lr']
        best_acc = 0.0
        val_loss_list = []
        for epoch in xrange(self.args['num_epoch']):
            n_train_batch = 0
            batch_size = self.args['batch_size']
            lr *= self.args['lr_decay']
            print >> self.out, '\033[1;36mThe epoch %d training: \033[0m' % epoch
            print >> lio, 'The epoch %d training: ' % epoch
            while True:
                prepared_x, prepared_y = self.get_batch('train', n_train_batch)
                if prepared_x is None:
                    break
                # print >> self.out, prepared_y
                feed = {self.lr: lr, self._x: prepared_x, self._y: prepared_y}
                _, loss = self.sess.run([self.train_step, self.loss_step], feed_dict=feed)
                if n_train_batch % 100 == 0:
                    print >> self.out, 'The iteration is %d train loss is: %f' % (n_train_batch, loss)
                    print >> lio, 'The iteration is %d train loss is: %f' % (n_train_batch, loss)
                if n_train_batch % 200 == 0:
                    loss, err = self.evaluate('test')
                    val_loss_list.append(loss)
                n_train_batch += 1
            _, acc = self.evaluate('test_new')
            acc = 1 - acc
            if acc > best_acc:
                best_acc = acc
            if acc > persistent_best_acc:
                persistent_best_acc = acc
                dirname = '../out/%.6f %s' % (acc, strnow())
                self.save(dirname)
                with open(os.path.join(dirname, 'params.log'), 'w') as f:
                    f.write(lio.getvalue())

        with open('../doc/val_loss_tf.csv', 'w') as f:
            for epoch, loss in zip(xrange(self.args['num_epoch']), val_loss_list):
                f.write('%.16f,%.16f\n' % (epoch, loss))
        print "miaomiaomiao"   
        print best_acc
        return best_acc

    def save(self, dirname):
        try:
            os.makedirs(dirname)
        except:
            pass
        saver = tf.train.Saver()
        return saver.save(self.sess, os.path.join(dirname, "model.ckpt"))


def run_single():
    global tf, tfnnutils, persistent_best_acc
    import tensorflow as tf
    import nnutils_tensorflow as tfnnutils
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
    try:
        acc = map(lambda s: safe_tofloat(s.split()[0]), os.listdir('../out'))
        acc.sort()
        persistent_best_acc = acc[-1]
    except:
        persistent_best_acc = 0
    print 'persistent_best_acc', persistent_best_acc
    data = data_process.read_all()
    model = Tone_Classification(data, args)
    t1 = time.time()
    model.build_model()
    t2 = time.time()
    acc = model.train()
    t3 = time.time()
    print 'build_model time %.16fs' % (t2 - t1)
    print 'train %s epoch time %.16fs' % (args['num_epoch'], t3 - t2)
    print >> lio, 'best acc', acc

    if not os.path.exists('../lio'):
        os.mkdir('../lio')
    with open('../lio/%.6f %s.log' % (acc, strnow()), 'w') as f:
        f.write(lio.getvalue())


if __name__ == "__main__":
    run_single()
