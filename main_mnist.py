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

# A function that shuffles a dataset, credits to https://github.com/uranusx86/BinaryNet-on-tensorflow
def shuffle(X,y):
    print(len(X))
    shuffle_parts = 1
    chunk_size = int(len(X)/shuffle_parts)
    shuffled_range = np.arange(chunk_size)

    X_buffer = np.copy(X[0:chunk_size])
    y_buffer = np.copy(y[0:chunk_size])

    for k in range(shuffle_parts):

        np.random.shuffle(shuffled_range)

        for i in range(chunk_size):

            X_buffer[i] = X[k*chunk_size+shuffled_range[i]]
            y_buffer[i] = y[k*chunk_size+shuffled_range[i]]

        X[k*chunk_size:(k+1)*chunk_size] = X_buffer
        y[k*chunk_size:(k+1)*chunk_size] = y_buffer

    return X,y

def train_epoch(inp, y, training, X, lab, sess, train_step, batch_size=100):
    batches = int(len(X)/batch_size)
    for i in range(batches):
        sess.run([train_step], feed_dict={inp:X[i*batch_size:(i+1)*batch_size], y: lab[i*batch_size:(i+1)*batch_size], training:True})


def main():
    batch_size = 100
    n_input = 28*28
    n_hidden = 2048
    n_output = 10
    drop_in = 0.2
    drop_hidden = 0.5
    epochs = 1000
    learning_rate_start = 3e-3
    learning_rate_end = 3e-7
    learning_rate_decay = (learning_rate_end/learning_rate_start)**(1./epochs)
    mnist_data = input_data.read_data_sets("MNIST_data/", one_hot=True)
    inp = tf.placeholder(tf.float32, [None, n_input])
    y = tf.placeholder(tf.float32, [None, n_output])
    training = tf.placeholder(tf.bool)
    g_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate_start, global_step = g_step, decay_steps = int(mnist_data.train.images.shape[0]/batch_size), decay_rate=learning_rate_decay)
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
    
    res = binn_mlp_mnist(inp, weights, biases, use_bias = True, training=training)
    cross_entropy = tf.square(tf.maximum(0., 1.-y*res))
    loss = tf.reduce_mean(cross_entropy)
    all_trainable_vars = [var for var in tf.trainable_variables()]
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss = loss, var_list=all_trainable_vars, global_step=g_step)
    correct_pred = tf.equal(tf.argmax(res, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.global_variables_initializer()
    sess= tf.Session()
    sess.run(init)

    old_acc = 0.0
    X_train, y_train = shuffle(mnist_data.train.images, mnist_data.train.labels)
    for i in range(epochs):
        train_epoch(inp, y, training, X_train, y_train, sess, train_step, batch_size)
        X_train, y_train = shuffle(mnist_data.train.images, mnist_data.train.labels)

        hist = sess.run([accuracy, loss],
                    feed_dict={
                        inp: mnist_data.test.images,
                        y: mnist_data.test.labels,
                        training: False
                    })
        print(hist)

        if hist[0] > old_acc:
            old_acc = hist[0]
            save_path = saver.save(sess, "./binn_model/model.ckpt")

if __name__ == '__main__':
    main()
