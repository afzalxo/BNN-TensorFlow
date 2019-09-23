import tensorflow as tf
import numpy as np
import time
import math
from tensorflow.examples.tutorials.mnist import input_data
from binn_mlp import binn_mlp_mnist

def one_hot_labels(labels, dimension=10):
    res = np.zeros((labels.shape[0], dimension))
    for i, seq in enumerate(labels):
        res[i, seq] = 1
    return res

def main():
    batch_size = 100
    n_input = 28*28
    n_hidden = 2048
    n_output = 10
    learning_rate = 3e-3
    drop_in = 0.2
    drop_hidden = 0.5
    iterations = 1000
    mnist_data = input_data.read_data_sets("MNIST_data/", one_hot=True)
    inp = tf.placeholder(tf.float32, [None, n_input])
    y = tf.placeholder(tf.float32, [None, n_output])
    weights = { #Glorot normal initializer, draws samples from a truncated normal distribution centered at 0 with std dev = sqrt(2/(fan_in + fan_out))
    'w0': tf.Variable(tf.truncated_normal([n_input, n_hidden], stddev=np.sqrt(2/(n_input+n_hidden)))), 
    'w1': tf.Variable(tf.truncated_normal([n_hidden, n_hidden], stddev=np.sqrt(2/(n_hidden+n_hidden)))), 
    'w2': tf.Variable(tf.truncated_normal([n_hidden, n_hidden], stddev=np.sqrt(2/(n_hidden+n_hidden)))),
    'w3': tf.Variable(tf.truncated_normal([n_hidden, n_output], stddev=np.sqrt(2/(n_hidden+n_output))))
    }
    biases = {
    'b0': tf.Variable(tf.constant(0.1, shape=[n_hidden])),
    'b1': tf.Variable(tf.constant(0.1, shape=[n_hidden])),
    'b2': tf.Variable(tf.constant(0.1, shape=[n_hidden])),
    'b3': tf.Variable(tf.constant(0.1, shape=[n_output]))
    }
    
    res = binn_mlp_mnist(inp, weights, biases, use_bias = True, training=True)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = res, labels = y)
    loss = tf.reduce_mean(cross_entropy)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss = cross_entropy)
    correct_pred = tf.equal(tf.argmax(res, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.global_variables_initializer()
    sess= tf.Session()
    sess.run(init)
    
    for ep in range(0, epochs):
        batch_x, batch_y = mnist_data.train.next_batch(batch_size)#train_x[ep*batch_size:(ep*batch_size+batch_size)], train_y[ep*batch_size:(ep*batch_size+batch_size)]
        sess.run(train_step, feed_dict={inp:batch_x, y:batch_y})
        if ep%100 == 0:
            b_loss, b_acc = sess.run([loss, accuracy], feed_dict={inp:batch_x, y:batch_y})
            print("Iteration: " +str(ep) + ", Train Loss = " + str(b_loss) + ", Train Acc = " + str(b_acc))
            test_b_x, test_b_y = mnist_data.test.next_batch(batch_size)
            test_loss, test_acc = sess.run([loss, accuracy], feed_dict={inp: test_b_x, y:test_b_y})
            print("Test Loss = " + str(test_loss) + ", Test Acc = " + str(test_acc))



if __name__ == '__main__':
    main()
