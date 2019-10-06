
import warnings
def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()
import numpy as np
import tensorflow as tf
import bnn_misc
from tensorflow.examples.tutorials.mnist import input_data



def main():
    model_path = './bnn_mnist_10ep.npz'
    arch = bnn_misc.BinNN(model_path)
    test_data = input_data.read_data_sets("MNIST_data/", one_hot=True).test
    inp = test_data.images[0:2] * 2 - 1
    for i in range(test_data.images.shape[0]):
        test_data.images[i] = test_data.images[i] * 2 - 1
    for i in range(test_data.labels.shape[0]):
        test_data.labels[i] = test_data.labels[i] * 2 - 1
        
    y = tf.placeholder(tf.float32, [None, 10])
    inp_placeholder = tf.placeholder(tf.float32, [None, 28*28])
    res = arch.build(inp_placeholder)
    loss = tf.reduce_mean(tf.square(tf.maximum(0., 1.-y*res)))
    correct_pred = tf.equal(tf.argmax(res, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    with tf.device('/gpu:0'):
        with tf.Session() as sess:
            #ret = sess.run(res, feed_dict={inp_placeholder:inp})
            #print(ret)
            hist = sess.run([accuracy, loss], feed_dict={inp_placeholder: test_data.images, y: test_data.labels})
            print(hist)

if __name__=='__main__':
    main()
