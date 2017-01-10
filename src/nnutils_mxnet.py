import numpy as np
import mxnet as mx


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
        self.W = theano.shared(np.empty((n_in, n_out)))
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
