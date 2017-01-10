import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool

'''
Random 
'''
rand_seed = 7
default_rng = np.random.RandomState(rand_seed)
default_srng = T.shared_randomstreams.RandomStreams(rand_seed)

def random_init(size, minval=-0.05, maxval=0.05, rng=None):
    if rng is None:
        rng = default_rng
    vals = rng.uniform(low=minval, high=maxval, size=size)
    return vals.astype(theano.config.floatX)

def const_init(size, value=0.1):
    return np.ones(size, dtype=theano.config.floatX) * value


'''
Activation Functions
'''
ReLU = lambda x: x * (x > 0)
sigmoid = T.nnet.sigmoid
tanh = T.tanh
softmax = T.nnet.softmax
linear = lambda x: x

class FCLayer(object):
    def __init__(self, n_in, n_out, act, has_bias=True):
        self.n_in = n_in
        self.n_out = n_out
        self.act = act
        self.has_bias = has_bias
        # self.W = theano.shared(np.empty((n_in, n_out)))
        self.W = theano.shared(random_init((self.n_in, self.n_out)))
        if self.has_bias:
            # self.b = theano.shared(np.empty(n_out))
            self.b = theano.shared(const_init(self.n_out))
        self.L1_Loss = abs(self.W).sum()
        self.L2_Loss = T.sum(T.sqr(self.W)) / 2
        # self.initialize()

    def initialize(self):
        self.W.set_value(random_init((self.n_in, self.n_out)))
        if self.has_bias:
            self.b.set_value(const_init(self.n_out))

    def forward(self, x):
        return self.act(T.dot(x, self.W) + self.b)

    @property
    def params(self):
        if self.has_bias:
            return [self.W, self.b]
        else:
            return [self.W]

    @params.setter
    def params(self, param_list):
        if self.has_bias:
            self.b.set_value(param_list[1].get_value())
        self.W.set_value(param_list[0].get_value())

class Dropout(object):

    instances = []

    def __init__(self, dprob, srng=None):
        self.dprob = dprob
        self.srng = srng if srng is not None else default_srng
        self.dropout_on = theano.shared(np.cast[theano.config.floatX](1.0), borrow=True)
        Dropout.instances.append(self)

    def forward(self, x):
        mask = self.srng.binomial(n=1, p=1-self.dprob, size=x.shape, dtype=theano.config.floatX)
        return x * self.dropout_on * mask + x * (1 - self.dprob) * (1 - self.dropout_on)

    @property
    def params(self):
        return None

    @staticmethod
    def set_dropout_on(training):
        if training:
            d_on = 1.0
        else:
            d_on = 0.0
        for ins in Dropout.instances:
            ins.dropout_on.set_value(d_on)

class Conv2D(object):
    def __init__(self, input_shape, filter_shape, act, zeropad='valid'):
        self.input_shape = input_shape
        self.filter_shape = filter_shape
        self.act = act
        self.zeropad = zeropad 

        W_vals = random_init(filter_shape)
        self.W = theano.shared(W_vals)
        b_vals = random_init((filter_shape[0],))
        self.b = theano.shared(b_vals)

    def forward(self, x):
        conv_out = conv2d(input=x, filters=self.W, filter_shape=self.filter_shape, input_shape=self.input_shape, border_mode=self.zeropad)
        return self.act(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))

    @property
    def params(self):
        return [self.W, self.b]

    @params.setter
    def params(self, param_list):
        self.W.set_value(param_list[0].get_value())
        self.b.set_value(param_list[1].get_value())

class AlphaGoFinalConv(object):
    def __init__(self, input_shape, filter_shape):
        self.input_shape = input_shape
        self.filter_shape = filter_shape
        assert(filter_shape[0] == 1)

        W_vals = random_init(filter_shape)
        self.W = theano.shared(W_vals)
        b_vals = random_init((input_shape[2] * input_shape[3],))
        self.b = theano.shared(b_vals)

    def forward(self, x):
        conv_out = conv2d(input=x, filters=self.W, filter_shape=self.filter_shape, input_shape=self.input_shape)
        conv_flatten = conv_out.flatten(2)
        return softmax(conv_flatten + self.b.dimshuffle('x', 0))

    @property
    def params(self):
        return [self.W, self.b]

    @params.setter
    def params(self, param_list):
        self.W.set_value(param_list[0].get_value())
        self.b.set_value(param_list[1].get_value())

class MaxPool2D(object):
    def __init__(self, pool_shape):
        self.pool_shape = pool_shape
    
    def forward(self, x):
        return pool.max_pool_2d(input=x, ds=self.pool_shape, ignore_border=True)

    @property
    def params(self):
        return None

class Flatten(object):
    def __init__(self, ndim):
        self.ndim = ndim
    
    def forward(self, x):
        return x.flatten(self.ndim)

    @property
    def params(self):
        return None