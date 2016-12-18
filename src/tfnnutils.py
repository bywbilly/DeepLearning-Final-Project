import tensorflow as tf
import numpy


def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
    return var


class FCLayer(object):
    def __init__(self, layer_name, n_in, n_out, act, has_bias=True):
        self.n_in = n_in
        self.n_out = n_out
        self.act = act
        self.has_bias = has_bias

        self.W = _variable_on_cpu('%s-W' % layer_name, [n_in, n_out],
                                  tf.random_uniform_initializer(minval=-0.05,maxval=0.05, dtype=tf.float32))
        self.b = _variable_on_cpu('%s-b' % layer_name, [n_out],
                                  #tf.random_uniform_initializer(minval=-0.05,maxval=0.05, dtype=tf.float32))
                                  tf.constant_initializer(0.1, dtype=tf.float32))
        self.L2_Loss = tf.nn.l2_loss(self.W)
        self.L1_Loss = tf.reduce_sum(tf.abs(self.W))

    def forward(self, x):
        if self.act is None:
            return tf.matmul(x, self.W) + self.b
        else:
            return self.act(tf.matmul(x, self.W) + self.b)


class Conv2D(object):
    def __init__(self, layer_name, filter_shape, strides=(1, 1, 1, 1), padding='SAME'):
        self.filter_shape = filter_shape
        self.strides = strides
        self.padding = padding

        self.W = _variable_on_cpu('%s-W' % layer_name, self.filter_shape,
                                  tf.random_uniform_initializer(minval=-0.05,maxval=0.05, dtype=tf.float32))
        self.b = _variable_on_cpu('%s-b' % layer_name, [self.filter_shape[3]],
                                  tf.random_uniform_initializer(minval=-0.05,maxval=0.05, dtype=tf.float32))
                                  #tf.constant_initializer(0.01, dtype=tf.float32))

    def forward(self, x):
        out = tf.nn.conv2d(input=x, filter=self.W, strides=(1, 1, 1, 1), padding=self.padding)
        out = tf.nn.bias_add(out, self.b)
        out = tf.nn.relu(out)
        return out


class MaxPool2D(object):
    def __init__(self, pool_size=2):
        self.ksize = (1, pool_size, pool_size, 1)

    def forward(self, x):
        return tf.nn.max_pool(value=x, ksize=self.ksize, strides=self.ksize, padding='VALID')


class Flatten(object):
    def forward(self, x):
        shape = x.get_shape().as_list()
        dim = numpy.prod(shape[1:])
        return tf.reshape(x, [shape[0], dim])
