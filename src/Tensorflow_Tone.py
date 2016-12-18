import numpy as np
import tensorflow as tf
import tfnnutils
import sys
import data_process
from pprint import pprint

class Tone_Classification():
    def __init__(self, data, args):
        self.input_dim = args['input_dim']
        self.hidden_dim = args['hidden_dim']
        self.batch_size = args['batch_size']
        self.init_lr = args['init_lr']
        data_process.shuffle(data)
        data_process.strip_zeros(data, 0.05)
        data_process.fix_length(data, self.input_dim//2, np.max)
        self.data_xs, self.data_ys = {}, {}
        for k, v in data.iteritems():
            xs, ys = [], []
            for datum in v:
                xs.append(datum.engy + datum.f0)
                ys.append(datum.tone)
            self.data_xs[k] = np.array(xs, np.float32)
            self.data_ys[k] = np.array(ys, np.int32) - 1
            print self.data_xs[k].shape, self.data_ys[k].shape

    def _loss(self, logits, L1_loss, L2_loss, labels):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits, labels, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        if args['use_L2']:
            cross_entropy_mean += args['L2_reg'] * L2_loss
        elif args['use_L1']:
            cross_entropy_mean += args['L1_reg'] * L1_loss
        return cross_entropy_mean

    def _forward(self, batch_x):
        layers = []
        dims = [self.input_dim] + self.hidden_dim
        for i in xrange(len(args['hidden_dim'])):
            layers.append(tfnnutils.FCLayer('FC%d'%(i+1), dims[i], dims[i+1], act=tf.nn.relu))

        L1_loss, L2_loss = 0., 0.
        for layer in layers:
            if hasattr(layer, 'L2_Loss'):
                L2_loss += layer.L2_Loss
            elif hasattr(layer, 'L1_Loss'):
                L1_loss += layer.L1_Loss
            batch_x = layer.forward(batch_x)
            #print batch_x
        
        pred = tf.nn.softmax(batch_x)
        #print pred
        
        return pred, batch_x, L1_loss, L2_loss
    
    def build_model(self):
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)
        self.lr = tf.placeholder(tf.float32, shape=[])
        #opt = tf.train.MomentumOptimizer(learning_rate = self.lr, momentum = 0.9)
        opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        # opt = tf.train.AdamOptimizer(learning_rate=self.lr)
        #opt = tf.train.AdagradOptimizer(learning_rate = self.lr)
        self._x = tf.placeholder(tf.float32, shape=[self.batch_size, self.input_dim])
        self._y = tf.placeholder(tf.int32)
        x = self._x
        y = self._y

        pred, logits, L1_loss, L2_loss = self._forward(x)
        loss = self._loss(logits, L1_loss, L2_loss, y)

        grads = opt.compute_gradients(loss)

        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        init = tf.initialize_all_variables()

        self.sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False))

        self.sess.run(init)
        tf.train.start_queue_runners(sess=self.sess)

        self.train_step = apply_gradient_op
        self.pred_step = pred
        self.loss_step = loss

    def predict(self, batch_x):
        feed = {self._x: batch_x}
        return self.sess.run(self.pred_step, feed_dict=feed)

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
        batch_size = args['batch_size']
        total_loss = 0.
        total_err = 0.
        n_batch = 0
        now_pos = 0
        while True:
            prepared_x, prepared_y = self.get_batch(dataset, n_batch)
            #print prepared_x, prepared_y
            if prepared_x is None:
                break
            feed = {self._x: prepared_x, self._y: prepared_y}
            loss, preds = self.sess.run([self.loss_step, self.pred_step], feed_dict=feed)
            total_loss += np.mean(loss)
            for i in range(len(preds)):
                if np.argmax(preds[i]) != prepared_y[i]:
                    # print 'preds[i] =', preds[i], 'prepared_y[i] =', prepared_y[i]
                    total_err += 1
            n_batch += 1
        loss = total_loss / n_batch
        err = total_err / (n_batch * batch_size)
        print 'evaluate %s: loss = %f err = \033[1;31m%f\033[0m' % (dataset, loss, err)
        return err

    def train(self, batch_x, batch_y):
        lr = self.init_lr
        last_err = 100
        cnt_hang = 0
        for epoch in xrange(20):
            n_train_batch = 0
            batch_size = args['batch_size']
            if epoch % 4 == 0:
                lr /= 2.0 
            print '\033[1;36mThe epoch %d training: \033[0m' % epoch
            while True:
                prepared_x, prepared_y = self.get_batch('train', n_train_batch)
                if prepared_x is None:
                    break
                # print prepared_y
                feed = {self.lr: lr, self._x: prepared_x, self._y: prepared_y}
                _, loss = self.sess.run([self.train_step, self.loss_step], feed_dict=feed)
                if n_train_batch % 100 == 0:
                    print 'The iteration is %d train loss is: %f' % (n_train_batch, loss)
                if n_train_batch % 200 == 0:
                    err = self.evaluate('test')
                    if abs(err - last_err) < 1e-3:
                        cnt_hang += 1
                        if cnt_hang >= 4:
                            return False
                    else:
                        cnt_hang = 0
                        last_err = err
                n_train_batch += 1
            self.evaluate('test_new')
        return True


if __name__ == "__main__":
    args = {
        'input_dim': 12,
        'hidden_dim': [12, 12, 4],
        'batch_size': 2,
        'init_lr': 0.0016,
        'use_L2': True,
        'use_L1': False,
        'L2_reg': 0.005,
        'L1_reg': 0.0005,
    }
    data = data_process.read_all()
    model = Tone_Classification(data, args)
    model.build_model()
    while not model.train(None, None):
        model.sess.run(tf.initialize_all_variables())
        print '\033[7mhung... auto restart...\033[0m'
