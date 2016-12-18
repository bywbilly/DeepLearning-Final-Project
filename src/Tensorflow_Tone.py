import numpy as np
import tensorflow as tf
import tfnnutils
import sys

#train_data = '/Users/bywbilly/DeepLearning/Final Project/toneclassifier/train_intergrate_'
#test_data = '/Users/bywbilly/DeepLearning/Final Project/toneclassifier/test_intergrate_'
#test_new_data = '/Users/bywbilly/DeepLearning/Final Project/toneclassifier/test_new_intergrate_'
train_data = '/Users/bywbilly/DeepLearning/Final Project/toneclassifier/train_intergrate_nozero_'
test_data = '/Users/bywbilly/DeepLearning/Final Project/toneclassifier/test_intergrate_nozero_'
test_new_data = '/Users/bywbilly/DeepLearning/Final Project/toneclassifier/test_new_intergrate_nozero_'

class Tone_Classification():
    def __init__(self, args):
        self.input_dim = args['input_dim']
        self.hidden_dim = args['hidden_dim']
        self.batch_size = args['batch_size']
        self.train_x = np.zeros((400, self.input_dim), np.float32)
        self.train_y = np.zeros((400, ), np.int64)
        self.test_x = np.zeros((40, self.input_dim), np.float32)
        self.test_y = np.zeros((40, ), np.float32)
        self.test_new_x = np.zeros((228, self.input_dim), np.float32)
        self.test_new_y = np.zeros((228, ), np.float32)
        self.output_dim = 4

    def _loss(self, logits, labels):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits, labels, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        return cross_entropy_mean
    def _forward(self, batch_x):
        layers = []
        for i in xrange(len(args['hidden_dim'])):
            if i == 0:
                layers.append(tfnnutils.FCLayer('FC%d' % (i + 1), self.input_dim, self.hidden_dim[i], act = tf.nn.relu))
            else:
                layers.append(tfnnutils.FCLayer('FC%d' % (i + 1), self.hidden_dim[i - 1], self.hidden_dim[i], act = tf.nn.relu))
        #layers.append(tfnnutils.FCLayer('FC1', self.input_dim, self.hidden_dim, act = tf.nn.relu)) 
        #layers.append(tfnnutils.FCLayer('FC2', self.hidden_dim, self.hidden_dim, act = tf.nn.relu)) 
        #layers.append(tfnnutils.FCLayer('FC3', self.hidden_dim, self.hidden_dim, act = tf.nn.relu)) 
        #layers.append(tfnnutils.FCLayer('FC4', self.hidden_dim, self.hidden_dim, act = tf.nn.relu)) 
        #layers.append(tfnnutils.FCLayer('FC5', self.hidden_dim, self.hidden_dim, act = tf.nn.relu)) 

        layers.append(tfnnutils.Flatten())
        for layer in layers:
            batch_x = layer.forward(batch_x)

        pred = tf.nn.softmax(batch_x)
        
        return pred, batch_x
    
    def build_model(self):
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)
        self.lr = tf.placeholder(tf.float32, shape = [])
        #opt = tf.train.MomentumOptimizer(learning_rate = self.lr, momentum = 0.9)
        #opt = tf.train.GradientDescentOptimizer(learning_rate = self.lr)
        opt = tf.train.AdamOptimizer(learning_rate = self.lr)
        self._x = tf.placeholder(tf.float32, shape = [self.batch_size, self.input_dim])
        self._y = tf.placeholder(tf.int64)
        x = self._x
        y = self._y

        pred, logits = self._forward(x)
        loss = self._loss(logits, y)

        grads = opt.compute_gradients(loss)

        apply_gradient_op = opt.apply_gradients(grads, global_step = global_step)

        init = tf.initialize_all_variables()

        self.sess = tf.Session(config = tf.ConfigProto(
            allow_soft_placement = True,
            log_device_placement = False))

        self.sess.run(init)
        tf.train.start_queue_runners(sess=self.sess)

        self.train_step = apply_gradient_op
        self.pred_step = pred
        self.loss_step = loss

    def predict(self, batch_x):
        feed = {self._x: batch_x}
        return self.sess.run(self.pred_step, feed_dict = feed)

    def prepared_test_data(self):
        f1 = open(test_data + 'f0', 'r')
        f2 = open(test_data + 'engy', 'r')
        flip = 0
        n = 0

        index1 = -1
        while True:
            if flip == 0:
                n = f1.readline().strip()
                if n == '':
                    print index1
                    break
                n = int(n)
                #n = int(f1.readline().strip())
                index1 += 1
                f2.readline()
                flip ^= 1
            else:
                block_size = int(2 * n / (args['input_dim'])) + 1
                cnt = 0
                flip ^= 1
                index2 = -1
                value1, value2 = 0.0, 0.0
                for i in xrange(n):
                    cnt += 1
                    value1 += float(f1.readline().strip())
                    value2 += float(f2.readline().strip())
                    if cnt == block_size or i == n - 1:
                        cnt = 0
                        index2 += 1
                        self.test_x[index1, index2] = value1 / (1.0 * block_size)
                        self.test_x[index1, index2 + args['input_dim'] / 2] = value2 / (1.0 * block_size)
                        value1, value2 = 0.0, 0.0 
                label = int(f1.readline().strip())
                f2.readline()
                self.test_y[index1] = label
        index = np.arange(40)
        np.random.shuffle(index)
        self.test_x[np.arange(40)] = self.test_x[index]
        self.test_y[np.arange(40)] = self.test_y[index]
        f1.close()
        f2.close()
        

    def prepared_train_data(self):
        f1 = open(train_data + 'f0', 'r')
        f2 = open(train_data + 'engy', 'r')
        flip = 0
        n = 0

        index1 = -1
        while True:
            if flip == 0:
                n = f1.readline().strip()
                if n == '':
                    print index1
                    break
                n = int(n)
                #n = int(f1.readline().strip())
                index1 += 1
                f2.readline()
                flip ^= 1
            else:
                block_size = int(2 * n / (args['input_dim'])) + 1
                cnt = 0
                flip ^= 1
                index2 = -1
                value1, value2 = 0.0, 0.0
                for i in xrange(n):
                    cnt += 1
                    value1 += float(f1.readline().strip())
                    value2 += float(f2.readline().strip())
                    #print i, value1, value2
                    if cnt == block_size or i == n - 1:
                        cnt = 0
                        index2 += 1
                        self.train_x[index1, index2] = value1 / (1.0 * block_size)
                        self.train_x[index1, index2 + args['input_dim'] / 2] = value2 / (1.0 * block_size)
                        value1, value2 = 0.0, 0.0
                label = int(f1.readline().strip())
                #print label
                #print "--------"
                f2.readline()
                self.train_y[index1] = label
        index = np.arange(400)
        np.random.shuffle(index)
        self.train_x[np.arange(400)] = self.train_x[index]
        self.train_y[np.arange(400)] = self.train_y[index]

        f1.close()
        f2.close()
    def prepared_test_new_data(self):
        f1 = open(test_new_data + 'f0', 'r')
        f2 = open(test_new_data + 'engy', 'r')
        flip = 0
        n = 0

        index1 = -1
        while True:
            if flip == 0:
                n = f1.readline().strip()
                if n == '':
                    print index1
                    break
                n = int(n)
                #n = int(f1.readline().strip())
                index1 += 1
                f2.readline()
                flip ^= 1
            else:
                block_size = int(2 * n / (args['input_dim'])) + 1
                cnt = 0
                flip ^= 1
                index2 = -1
                value1, value2 = 0.0, 0.0
                for i in xrange(n):
                    cnt += 1
                    value1 += float(f1.readline().strip())
                    value2 += float(f2.readline().strip())
                    #print i, value1, value2
                    if cnt == block_size or i == n - 1:
                        cnt = 0
                        index2 += 1
                        self.test_new_x[index1, index2] = value1 / (1.0 * block_size)
                        self.test_new_x[index1, index2 + args['input_dim'] / 2] = value2 / (1.0 * block_size)
                        value1, value2 = 0.0, 0.0
                label = int(f1.readline().strip())
                #print label
                #print "--------"
                f2.readline()
                self.test_new_y[index1] = label
        index = np.arange(228)
        np.random.shuffle(index)
        self.test_new_x[np.arange(228)] = self.test_new_x[index]
        self.test_new_y[np.arange(228)] = self.test_new_y[index]

        f1.close()
        f2.close()
    def get_test_batch(self, index):
        ret_x = np.zeros((self.batch_size, self.input_dim), np.float32)
        ret_y = np.zeros((self.batch_size, ), np.int64)

        if ((index + 1) * self.batch_size - 1 > self.test_x.shape[0]):
            return None, None

        st = index * self.batch_size
        ed = st + self.batch_size
        ret_x = self.test_x[st:ed]
        ret_y = self.test_y[st:ed]
        return ret_x, ret_y

    def get_test_new_batch(self, index):
        ret_x = np.zeros((self.batch_size, self.input_dim), np.float32)
        ret_y = np.zeros((self.batch_size, ), np.int64)

        if ((index + 1) * self.batch_size - 1 > self.test_new_x.shape[0]):
            return None, None

        st = index * self.batch_size
        ed = st + self.batch_size
        ret_x = self.test_new_x[st:ed]
        ret_y = self.test_new_y[st:ed]
        return ret_x, ret_y

    def get_train_batch(self, index):
        ret_x = np.zeros((self.batch_size, self.input_dim), np.float32)
        ret_y = np.zeros((self.batch_size, ), np.int64)
        
        if ((index + 1) * self.batch_size - 1 > self.train_x.shape[0]):
            return None, None
        st = index * self.batch_size
        ed = st + self.batch_size
        ret_x = self.train_x[st:ed]
        ret_y = self.train_y[st:ed]
        return ret_x, ret_y

    def evaluate(self):
        batch_size = args['batch_size']
        total_loss = 0.
        total_err = 0.
        n_batch = 0
        now_pos = 0
        print 'evaluating...'
        while True:
            prepared_x, prepared_y = self.get_test_batch(n_batch)
            #print prepared_x, prepared_y
            if prepared_x is None:
                break
            feed = {self._x: prepared_x, self._y: prepared_y}
            loss, preds = self.sess.run([self.loss_step, self.pred_step], feed_dict = feed)
            total_loss += np.mean(loss)
            for i in range(len(preds)):
                if np.argmax(preds[i]) != prepared_y[i]:
                    total_err += 1
            n_batch += 1
        print 'loss = %f err = %f' % (total_loss / n_batch, total_err / (n_batch * batch_size))
    
    def evaluate_new(self):
        batch_size = args['batch_size']
        total_loss = 0.
        total_err = 0.
        n_batch = 0
        now_pos = 0
        print 'evaluating...'
        while True:
            prepared_x, prepared_y = self.get_test_new_batch(n_batch)
            #print prepared_x, prepared_y
            if prepared_x is None:
                break
            feed = {self._x: prepared_x, self._y: prepared_y}
            loss, preds = self.sess.run([self.loss_step, self.pred_step], feed_dict = feed)
            total_loss += np.mean(loss)
            for i in range(len(preds)):
                if np.argmax(preds[i]) != prepared_y[i]:
                    total_err += 1
            n_batch += 1
        print 'Test the noise data:'
        print '!!loss = %f err = %f' % (total_loss / n_batch, total_err / (n_batch * batch_size))


    def train(self, batch_x, batch_y):
        lr = 0.001

        for epoch in xrange(15):
            n_train_batch = 0
            batch_size = args['batch_size']
            if epoch % 3 == 0:
                lr /= 2.0 
            print 'The epoch %d training: ' % epoch
            while True:
                prepared_x, prepared_y = self.get_train_batch(n_train_batch)
                if prepared_x is None:
                    break
                #print prepared_y
                feed = {self.lr: lr, self._x: prepared_x, self._y: prepared_y}
                _, loss = self.sess.run([self.train_step, self.loss_step], feed_dict=feed)
                if n_train_batch % 100 == 0:
                    print 'The iteration is %d train loss is: %f' % (n_train_batch, loss)
                if n_train_batch % 200 == 0:
                    self.evaluate()
                n_train_batch += 1
            self.evaluate_new()


if __name__ == "__main__":

    args = {}
    args['input_dim'] = 20 
    #args['hidden_dim'] = 60
    args['hidden_dim'] = [60, 40, 40, 20]
    args['batch_size'] = 4
    model = Tone_Classification(args)
    model.prepared_train_data()
    model.prepared_test_data()
    model.prepared_test_new_data()
    model.build_model()
    model.train(None, None)
