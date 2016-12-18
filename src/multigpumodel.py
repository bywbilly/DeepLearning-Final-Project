import argparse
import os
import sys
import time
from multiprocessing import Queue, Process

import tensorflow as tf
from ioutils import produce
import numpy

import tfnnutils
import trai
from trai.tfmodel import AbstractTFModel
from trai.tfmodel import TFModelHandler

N_FEATURE=77
N_CLASS=2


def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
    return var


class KGSReader(object):
    def __init__(self, path, batch_size):
        self.queue = Queue(16)
        self.p = Process(target=produce, args=(path, self.queue, batch_size, batch_size * 16))

    def start(self):
        self.p.start()

    def get_batch(self):
        return self.queue.get()

    def join(self):
        self.p.join()


class TFMultiGPUModel(AbstractTFModel):
    def __init__(self, args):
        AbstractTFModel.__init__(self, model_name='TFMultiGPUModel')
        self.args = args

    def _loss(self, logits, labels):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits, labels, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        return cross_entropy_mean

    # gpu
    def _forward(self, batch_x):
        n_filter = 128
        layers = list()
        layers.append(tfnnutils.Conv2D(layer_name='conv1', filter_shape=(5, 5, N_FEATURE, n_filter)))
        layers.append(tfnnutils.Conv2D(layer_name='conv2', filter_shape=(3, 3, n_filter, n_filter)))
        layers.append(tfnnutils.Conv2D(layer_name='conv3', filter_shape=(3, 3, n_filter, n_filter)))
        layers.append(tfnnutils.Conv2D(layer_name='conv4', filter_shape=(3, 3, n_filter, n_filter)))
        layers.append(tfnnutils.Conv2D(layer_name='conv5', filter_shape=(3, 3, n_filter, n_filter)))
        layers.append(tfnnutils.Conv2D(layer_name='conv6', filter_shape=(3, 3, n_filter, n_filter)))
        layers.append(tfnnutils.Conv2D(layer_name='conv7', filter_shape=(3, 3, n_filter, n_filter)))
        layers.append(tfnnutils.Conv2D(layer_name='conv8', filter_shape=(3, 3, n_filter, n_filter)))
        layers.append(tfnnutils.Conv2D(layer_name='conv9', filter_shape=(3, 3, n_filter, n_filter)))
        layers.append(tfnnutils.Conv2D(layer_name='conv10', filter_shape=(1, 1, n_filter, 1)))

        layers.append(tfnnutils.Flatten())
        # layers.append(tfnnutils.FCLayer(layer_name='logit', n_in=19 * 19 * 1, n_out=361, act=None))
        for layer in layers:
            batch_x = layer.forward(batch_x)

        pred = tf.nn.softmax(batch_x)
        return pred, batch_x

    def _average_gradients(self, tower_grads):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(0, grads)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)

        return average_grads

    def build_model(self):
        with tf.Graph().as_default(), tf.device('/cpu:0'):
            global_step = tf.get_variable(
                'global_step', [],
                initializer=tf.constant_initializer(0), trainable=False)
            lr = 0.01
            opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)

            self._x = tf.placeholder(tf.float32, shape=[args.batch_size, 19, 19, N_FEATURE])
            self._y = tf.placeholder(tf.int64)
            x = self._x
            y = self._y

            split_x = tf.split(0, args.num_gpu, x)
            split_y = tf.split(0, args.num_gpu, y)

            tower_grads = []
            tower_preds = []
            tower_losses = []

            for i in range(0, self.args.num_gpu):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('sl_%d' % (i)) as scope:
                        pred, logits = self._forward(split_x[i])
                        loss_on_tower = self._loss(logits, split_y[i])

                        tower_losses.append(loss_on_tower)
                        grads = opt.compute_gradients(loss_on_tower)
                        tf.get_variable_scope().reuse_variables()
                        tower_grads.append(grads)
                        tower_preds.append(pred)

            # We must calculate the mean of each gradient. Note that this is the
            # synchronization point across all towers.
            grads = self._average_gradients(tower_grads)
            apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

            init = tf.initialize_all_variables()

            self.sess = tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False))
            self.sess.run(init)
            tf.train.start_queue_runners(sess=self.sess)

            self.saver = tf.train.Saver(tf.all_variables())

            self.train_step = apply_gradient_op
            self.pred_step = tf.reshape(tower_preds, [args.batch_size, 361])
            self.loss_step = tower_losses

    def predict(self, batch_x):
        feed = {self._x: batch_x}
        return self.sess.run(self.pred_step, feed_dict=feed)

    def evaluate(self):
        batch_size = args.batch_size
        test_reader = KGSReader(args.test_filename, batch_size)
        test_reader.start()
        start_time = time.time()
        total_loss = 0.
        total_err = 0.
        n_batch = 0
        print 'evaluating...'
        while True:
            if n_batch == 100:
                break
            prepared_x, prepared_y = test_reader.get_batch()
            if prepared_x is None:
                break
            feed = {self._x: prepared_x, self._y: prepared_y}
            loss, preds = self.sess.run([self.loss_step, self.pred_step], feed_dict=feed)
            total_loss += numpy.mean(loss)
            for i in range(len(preds)):
                if numpy.argmax(preds[i]) == prepared_y[i]:
                    total_err += 1
            n_batch += 1
        print 'evaluating done in %f' % (time.time() - start_time)
        print 'loss=%f err=%f' % (total_loss / n_batch, total_err / (n_batch * batch_size))

    def train(self, batch_x, batch_y):
        n_train_batch = 0

        #self.saver.restore(self.sess, args.load_model)
        for epoch in range(5):
            batch_size = args.batch_size
            train_reader = KGSReader(args.train_filename, batch_size)
            train_reader.start()
            while True:
                start_time = time.time()
                prepared_x, prepared_y = train_reader.get_batch()
                if prepared_x is None:
                    break
                feed = {self._x: prepared_x, self._y: prepared_y}
                _, loss = self.sess.run([self.train_step, self.loss_step], feed_dict=feed)
                if n_train_batch % 100 == 0:
                    print 'train loss on GPUs:' + str(loss)
                    duration = time.time() - start_time
                    num_examples_per_step = args.batch_size * args.num_gpu
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = duration / args.num_gpu
                    format_str = ('%.1f examples/sec; %.3f sec/batch')
                    print (format_str % (examples_per_sec, sec_per_batch))
                if n_train_batch % 100000 == 0:
                    self.evaluate()
                    self.saver.save(self.sess, args.save_model_path, global_step=n_train_batch)
                n_train_batch += 1

            train_reader.join()


class TFLeNetModelService(trai.ModelService):
    def __init__(self, args, model_name):
        trai.ModelService.__init__(self, args)
        handler = TFModelHandler(args, model_name)
        self.set_handler(handler)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument('--save_model_path', type=str,
                           default='/home/dapt/project/ai-platform/yi/go-models/tf-model/sl_policy.ckpt')
    argparser.add_argument('--load_model', type=str,
                           default='/home/dapt/project/ai-platform/yi/go-models/tf-model/sl_policy-10000.ckpt')
    argparser.add_argument('--batch_size', type=int, default=128)
    argparser.add_argument('--n_filter', type=int, default=64)

    argparser.add_argument('--port', type=int, default=50051)
    argparser.add_argument('--num_gpu', type=int, default=2)
    argparser.add_argument('--train_filename', type=str, default='/home/dapt/kgsgo-train.dat')
    argparser.add_argument('--test_filename', type=str, default='/home/dapt/kgsgo-test.dat')

    args = argparser.parse_args()

    # service = TFLeNetModelService(args, TFMultiGPUModel)
    # service.start()

    model = TFMultiGPUModel(args)
    model.build_model()

    model.train(None, None)
