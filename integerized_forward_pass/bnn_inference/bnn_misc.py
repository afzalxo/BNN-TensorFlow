import tensorflow as tf
import numpy as np

class BinNN:
    def __init__(self, model_path):
        self.model = np.load(model_path)
        for x in self.model.files:
            print(x + " " + str(self.model[x].shape))
        print(self.model['l1_b'][0:50])

    #Implements y = +1 for x > 0 and y = -1 for x <= 0
    def sign_binarize(self, inp):
        h_sig = tf.clip_by_value((inp+1.)/2., 0, 1)
        round_out = tf.round(h_sig)
        round_fin = h_sig + tf.stop_gradient(round_out - h_sig)
        return (2.*round_fin - 1.)

    def bin_dense_layer(self, in_act, in_w, in_b, name='Bin_Dense_L'):
        bin_w = self.sign_binarize(in_w)
        res = tf.matmul(in_act, bin_w)
        res = tf.nn.bias_add(res, self.sign_binarize(in_b))
        return res 

    def build(self, in_act):
        in_act = tf.contrib.layers.flatten(in_act)
        layer0_dense = self.bin_dense_layer(in_act, self.model['l0_w'], self.model['l0_b'], name='layer0_dense')
        layer0_bn = tf.nn.batch_normalization(layer0_dense, mean=self.model['l0_mean'], variance=self.model['l0_variance'], offset=self.model['l0_beta'], scale=self.model['l0_gamma'], variance_epsilon=1e-4, name='layer0_bn')
        layer0_sig = self.sign_binarize(layer0_bn)
        #return layer0_sig

        layer1_dense = self.bin_dense_layer(layer0_sig, self.model['l1_w'], self.model['l1_b'], name='layer1_dense')
        layer1_bn = tf.nn.batch_normalization(layer1_dense, mean=self.model['l1_mean'], variance=self.model['l1_variance'], offset=self.model['l1_beta'], scale=self.model['l1_gamma'], variance_epsilon=1e-4, name='layer1_bn')
        layer1_sig = self.sign_binarize(layer1_bn)

        layer2_dense = self.bin_dense_layer(layer1_sig, self.model['l2_w'], self.model['l2_b'], name='layer2_dense')
        layer2_bn = tf.nn.batch_normalization(layer2_dense, mean=self.model['l2_mean'], variance=self.model['l2_variance'], offset=self.model['l2_beta'], scale=self.model['l2_gamma'], variance_epsilon=1e-4, name='layer2_bn')
        layer2_sig = self.sign_binarize(layer2_bn)

        layer3_dense = self.bin_dense_layer(layer2_sig, self.model['l3_w'], self.model['l3_b'], name='layer3_dense')
        layer3_bn = tf.nn.batch_normalization(layer3_dense, mean=self.model['l3_mean'], variance=self.model['l3_variance'], offset=self.model['l3_beta'], scale=self.model['l3_gamma'], variance_epsilon=1e-4, name='layer3_bn')
        return layer3_bn

'''
inp = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],[-2, 0, -8, -9, 3, 9, -2, 9, 0.25, 0.75, -0.25, -0.75]], dtype=np.float32)
inp1 = tf.placeholder(dtype=tf.float32, shape=(2, 12))
res = bin_dense_layer(inp1, 2048)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    ret=sess.run(res, feed_dict={inp1:inp})
    print(ret)
'''
