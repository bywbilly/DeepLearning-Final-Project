import numpy as np
import copy
import itertools
import os
import sys
import time
import multiprocessing
import data_process
from pprint import pprint

class Tone_Classification():
    def __init__(self, data, args, verbose=True):
        self.input_dim = args['input_dim']
        self.hidden_dim = args['hidden_dim']
        self.batch_size = args['batch_size']
        self.args = args
        self.out = sys.stdout if verbose else open(os.devnull, 'w')
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
        dims = [self.input_dim] + self.hidden_dim
        for i in xrange(len(self.hidden_dim)):
            layers.append(tfnnutils.FCLayer('FC%d'%(i+1), dims[i], dims[i+1], act=tf.nn.relu))

        L1_loss, L2_loss = 0., 0.
        for layer in layers:
            if hasattr(layer, 'L2_Loss'):
                L2_loss += layer.L2_Loss
            elif hasattr(layer, 'L1_Loss'):
                L1_loss += layer.L1_Loss
            batch_x = layer.forward(batch_x)
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

        init = tf.global_variables_initializer()

        config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=False)
        config.gpu_options.allow_growth=True
        self.sess = tf.Session(config=config)

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
        return err

    def train(self, stop_if_hang=True):
        lr = self.args['init_lr']
        last_err = 100
        cnt_hang = 0
        for epoch in xrange(20):
            n_train_batch = 0
            batch_size = self.args['batch_size']
            if epoch % 4 == 0:
                lr /= 2.0 
            print >> self.out, '\033[1;36mThe epoch %d training: \033[0m' % epoch
            while True:
                prepared_x, prepared_y = self.get_batch('train', n_train_batch)
                if prepared_x is None:
                    break
                # print >> self.out, prepared_y
                feed = {self.lr: lr, self._x: prepared_x, self._y: prepared_y}
                _, loss = self.sess.run([self.train_step, self.loss_step], feed_dict=feed)
                if n_train_batch % 100 == 0:
                    print >> self.out, 'The iteration is %d train loss is: %f' % (n_train_batch, loss)
                if n_train_batch % 200 == 0:
                    err = self.evaluate('test')
                    if abs(err - last_err) < 1e-3:
                        cnt_hang += 1
                        if cnt_hang >= 4 and stop_if_hang:
                            return False
                    else:
                        cnt_hang = 0
                        last_err = err
                n_train_batch += 1
        return True

    def save(self, dirname):
        try:
            os.makedirs(dirname)
        except:
            pass
        saver = tf.train.Saver()
        return saver.save(self.sess, os.path.join(dirname, "model.ckpt"))


def run_single():
    global tf, tfnnutils
    import tensorflow as tf
    import nnutils_tensorflow as tfnnutils
    args = {
        'input_dim': 12,
        'hidden_dim': [12, 12, 4],
        'batch_size': 2,
        'optimizer': 'sgd',
        'init_lr': 0.0016,
        'use_L2': True,
        'use_L1': False,
        'L2_reg': 0.005,
        'L1_reg': 0.0005,
    }
    data = data_process.read_all()
    model = Tone_Classification(data, args)
    model.build_model()
    while not model.train():
        model.sess.run(tf.global_variables_initializer())
        print '\033[7mhung... auto restart...\033[0m'
    model.evaluate('test_new')


def _grid_search_init():
    global alldata, best_err, tf, tfnnutils
    import tensorflow as tf
    import nnutils_tensorflow as tfnnutils
    alldata = data_process.read_all()
    best_err = 1e2

def _grid_search_worker(args):
    global alldata, best_err
    err, path = 1e2, None
    for i in xrange(3):
        data = copy.deepcopy(alldata)
        tf.reset_default_graph()
        model = Tone_Classification(data, args, verbose=False)
        model.build_model()
        model.train(stop_if_hang=False)
        now_err = model.evaluate('test_new')
        err = min(err, now_err)
        if now_err < best_err:
            best_err = now_err
            path = model.save('../out/%.6f' % err)
    return args, err, path

def run_grid_search(num_worker):
    g_input_dim = [8, 12, 30]
    g_dims = [2, 3]
    g_dim_size = [10, 20, 40, 80]
    g_batch_size = [1, 2, 4, 8]
    g_optimizer = ['sgd', 'adam']
    g_lr = [0.0008, 0.0016, 0.0064]
    g_l2 = [0, 0.005]

    tasks = []
    for input_dim, dims, batch_size, optimizer, lr, l2 in itertools.product(g_input_dim, g_dims, g_batch_size, g_optimizer, g_lr, g_l2):
        for hidden_dim in itertools.product(*itertools.repeat(g_dim_size, dims)):
            tasks.append({
                'input_dim': input_dim,
                'hidden_dim': list(hidden_dim),
                'batch_size': batch_size,
                'optimizer': optimizer,
                'init_lr': lr,
                'use_L2': bool(l2),
                'use_L1': False,
                'L2_reg': l2,
                'L1_reg': .0
            })

    pool = multiprocessing.Pool(processes=num_worker, initializer=_grid_search_init)
    best = 1e2
    st = time.time()
    try:
        for done, (args, err, path) in enumerate(pool.imap_unordered(_grid_search_worker, tasks), start=1):
            dur = time.time() - st
            rhh, rmm, rss = dur // 3600, dur // 60 % 60, dur % 60
            dur = (len(tasks) - done) * dur / done
            ehh, emm, ess = dur // 3600, dur // 60 % 60, dur % 60
            sys.stderr.write('done %d/%d   %dh%02dm%02ds elapsed   %dh%02dm%02ds eta\n' % (done, len(tasks), rhh, rmm, rss, ehh, emm, ess))
            sys.stderr.flush()
            if err < best:
                best = err
                print 'new best!', 'err', err, 'saved at ', path
            str = ['err', repr(err)]
            for k in sorted(args.iterkeys()):
                str.append(k)
                str.append(repr(args[k]))
            print ' '.join(str)
            sys.stdout.flush()
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
        raise


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'grid':
        run_grid_search(int(sys.argv[2]))
    else:
        run_single()
