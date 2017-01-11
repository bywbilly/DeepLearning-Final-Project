import tensorflow as tf
import numpy


def _variable(name, shape, initializer):
    var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
    return var


class InputLayer(object):
    def __init__(self):
        self.shape = None

    def __str__(self):
        return 'InputLayer() shape: %r' % (self.shape, )

    def forward(self, _, x):
        self.shape = x.get_shape().as_list()
        return x


class FCLayer(object):
    def __init__(self, layer_name, n_out, act, has_bias=True):
        self.layer_name = layer_name
        self.n_out = n_out
        self.act = act
        self.has_bias = has_bias
        self.shape = None

    def __str__(self):
        return 'FCLayer(layer_name=%s, n_out=%d, act=%r) shape: %r' % (self.layer_name, self.n_out, self.act, self.shape)

    def forward(self, last_layer, x):
        self.W = _variable('%s-W' % self.layer_name, [last_layer.shape[1], self.n_out],
                           tf.random_uniform_initializer(minval=-0.05,maxval=0.05, dtype=tf.float32))
        self.b = _variable('%s-b' % self.layer_name, [self.n_out],
                           # tf.random_uniform_initializer(minval=-0.05,maxval=0.05, dtype=tf.float32))
                           tf.constant_initializer(0.1, dtype=tf.float32))
        self.L2_Loss = tf.nn.l2_loss(self.W)
        self.L1_Loss = tf.reduce_sum(tf.abs(self.W))

        out = tf.matmul(x, self.W) + self.b
        if self.act is not None:
            out = self.act(out)
        self.shape = out.get_shape().as_list()
        return out


class Conv2D(object):
    def __init__(self, layer_name, ksize, kernels, strides=(1, 1), padding='SAME'):
        self.layer_name = layer_name
        self.ksize = ksize
        self.kernels = kernels
        self.strides = strides
        self.padding = padding
        self.shape = None

    def __str__(self):
        return 'Conv2D(layer_name=%s, ksize=%r, kernels=%d, strides=%r, padding=%r) shape: %r' % (self.layer_name, self.ksize, self.kernels, self.strides, self.padding, self.shape)

    def forward(self, last_layer, x):
        filter_shape = (self.ksize[0], self.ksize[1], last_layer.shape[3], self.kernels)
        self.W = _variable('%s-W' % self.layer_name, filter_shape,
                           tf.random_uniform_initializer(minval=-0.05,maxval=0.05, dtype=tf.float32))
        self.b = _variable('%s-b' % self.layer_name, [self.kernels],
                           #tf.random_uniform_initializer(minval=-0.05,maxval=0.05, dtype=tf.float32))
                           tf.constant_initializer(0.1, dtype=tf.float32))

        out = tf.nn.conv2d(input=x, filter=self.W, strides=(1, self.strides[0], self.strides[1], 1), padding=self.padding)
        out = tf.nn.bias_add(out, self.b)
        out = tf.nn.relu(out)
        self.shape = out.get_shape().as_list()
        return out

class MaxPool(object):
    def __init__(self, ksize=(2, 2), padding = 'VALID'):
        self.ksize = ksize
        self.padding = padding
        self.shape = None

    def __str__(self):
        return 'MaxPool(ksize=%r, padding=%r) shape: %r' % (self.ksize, self.padding, self.shape)

    def forward(self, last_layer, x):
        ksize = (1, self.ksize[0], self.ksize[1], 1)
        out = tf.nn.max_pool(x, ksize, ksize, self.padding)
        self.shape = out.get_shape().as_list()
        return out


class Dropout(object):
    pkeep = None

    def __init__(self):
        self.shape = None
        if Dropout.pkeep is None:
            Dropout.pkeep = tf.placeholder(tf.float32)

    def __str__(self):
        return 'Dropout() shape: %r' % (self.shape, )

    def forward(self, last_layer, x):
        out = tf.nn.dropout(x, Dropout.pkeep)
        self.shape = out.get_shape().as_list()
        return out


class Flatten(object):
    def __init__(self):
        self.shape = None

    def __str__(self):
        return 'Flatten() shape: %r' % (self.shape)

    def forward(self, last_layer, x):
        shape = x.get_shape().as_list()
        dim = numpy.prod(shape[1:])
        out = tf.reshape(x, [shape[0], dim])
        self.shape = out.get_shape().as_list()
        return out
