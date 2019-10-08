import tensorflow as tf
import numpy as np
import time
import sys
import math
from tensorflow.examples.tutorials.mnist import input_data
from bnn_mlp import binn_mlp_mnist
from bnn_misc import compute_gradients
import matplotlib.pyplot as plt

def one_hot_labels(labels, dimension=10):
    res = np.zeros((labels.shape[0], dimension))
    for i, seq in enumerate(labels):
        res[i, seq] = 1
    return res

# A function that shuffles a dataset, credits to https://github.com/uranusx86/BinaryNet-on-tensorflow
def shuffle(X,y):
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

def train_epoch(inp, y, training, acc, lo, X, lab, sess, train_bn_step, batch_size=100):
    batches = int(len(X)/batch_size)
    for i in range(batches):
        hist0 = sess.run([acc, lo, train_bn_step], feed_dict={inp:X[i*batch_size:(i+1)*batch_size], y: lab[i*batch_size:(i+1)*batch_size], training:True})
        print(hist0[0], hist0[1])
def main():
    batch_size = 100
    n_input = 28*28
    n_hidden = 4096
    n_output = 10
    drop_in = 0.2
    drop_hidden = 0.5
    epochs = 10
    learning_rate_start = 3e-3
    learning_rate_end = 3e-5
    learning_rate_decay = (learning_rate_end/learning_rate_start)**(1./epochs)
    lr_mod_start = 1.e-0
    lr_mod_end = 1.e-3
    lr_mod_decay = (lr_mod_end/lr_mod_start)**(1./epochs)
    mnist_data = input_data.read_data_sets("MNIST_data/", one_hot=True)
    for i in range(mnist_data.train.images.shape[0]):
        mnist_data.train.images[i] = mnist_data.train.images[i] * 2 - 1
    for i in range(mnist_data.test.images.shape[0]):
        mnist_data.test.images[i] = mnist_data.test.images[i] * 2 - 1
    for i in range(mnist_data.train.labels.shape[0]):
        mnist_data.train.labels[i] = mnist_data.train.labels[i] * 2 - 1
    for i in range(mnist_data.test.labels.shape[0]):
        mnist_data.test.labels[i] = mnist_data.test.labels[i] * 2 - 1
        
    inp = tf.placeholder(tf.float32, [None, n_input])
    y = tf.placeholder(tf.float32, [None, n_output])
    training = tf.placeholder(tf.bool)
    g_step_kern = tf.Variable(0, trainable=False)
    g_step_mod = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate_start, global_step = g_step_kern, decay_steps = int(mnist_data.train.images.shape[0]/batch_size), decay_rate=learning_rate_decay)
    lr_mod = tf.train.exponential_decay(lr_mod_start, global_step = g_step_mod, decay_steps = int(mnist_data.train.images.shape[0]/batch_size), decay_rate=lr_mod_decay)
    res = binn_mlp_mnist(inp, use_bias = True, training=training)
    cross_entropy = tf.square(tf.maximum(0., 1.-y*res))
    loss = tf.reduce_mean(cross_entropy)
    all_trainable_vars = [var for var in tf.trainable_variables()]# if not var.name.endswith('modulo:0')]
    #train_mod_vars = [var for var in tf.trainable_variables() if var.name.endswith('modulo:0')]
    print("--All Trainable Vars------------------------>>>>>>>")
    print(all_trainable_vars)
    print("--End All Trainable Vars-------------------->>>>>>>")
    print("--All Global Vars------------------------>>>>>>>")
    print([var for var in tf.global_variables()])
    print("--End All Global Vars-------------------->>>>>>>")
    update_operations = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    print("--Update Ops-------------------------------->>>>>>>")
    print(update_operations)
    print("--End Update Ops---------------------------->>>>>>>")
    with tf.control_dependencies(update_operations):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        #optimizer_mod = tf.train.AdamOptimizer(lr_mod)
        #grad_w = optimizer.compute_gradients(loss = loss, var_list = all_trainable_vars)
        #train_bn_step = optimizer.apply_gradients(grad_w, global_step = g_step_kern)
        #grad_m = optimizer_mod.compute_gradients(loss = loss, var_list = train_mod_vars)
        #train_mod_step = optimizer_mod.apply_gradients(grad_m, global_step = g_step_mod)
        train_bn_step = optimizer.minimize(loss = loss, var_list=all_trainable_vars, global_step=g_step_kern)
    correct_pred = tf.equal(tf.argmax(res, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.global_variables_initializer()
    sess= tf.Session()
    sess.run(init)
    saver = tf.train.Saver()

    old_acc = 0.0
    store_epoch = 0
    X_train, y_train = shuffle(mnist_data.train.images, mnist_data.train.labels)
    t_start = time.time()
    for i in range(epochs):
        train_epoch(inp, y, training, accuracy, loss, X_train, y_train, sess, train_bn_step, batch_size)
        X_train, y_train = shuffle(mnist_data.train.images, mnist_data.train.labels)

        hist = sess.run([accuracy, loss],
                    feed_dict={
                        inp: mnist_data.test.images,
                        y: mnist_data.test.labels,
                        training: False
                    })
        print("Epoch %d, Test Acc: %f, Loss %f, Current Best Acc: %f" % (i, hist[0], hist[1], old_acc))
        if hist[0] > old_acc:
            net_params = sess.run(tf.global_variables())
            net_params = net_params[2:]
            np.savez('bnn_mnist_10ep.npz', l0_w=net_params[0], l0_b=net_params[1], l0_gamma=net_params[2], l0_beta=net_params[3], l0_mean=net_params[4], l0_variance=net_params[5], l1_w=net_params[6], l1_b=net_params[7], l1_gamma=net_params[8], l1_beta = net_params[9], l1_mean=net_params[10], l1_variance=net_params[11], l2_w=net_params[12],l2_b=net_params[13], l2_gamma=net_params[14], l2_beta=net_params[15], l2_mean = net_params[16], l2_variance=net_params[17],l3_w=net_params[18],l3_b=net_params[19], l3_gamma=net_params[20], l3_beta=net_params[21], l3_mean = net_params[22], l3_variance=net_params[23])
            old_acc = hist[0]
            store_epoch = i
            save_path = saver.save(sess, "./binn_model/model.ckpt")
    t_end = time.time()
    np.set_printoptions(edgeitems=500)
    reto = sess.run(res, feed_dict={inp: mnist_data.test.images[0:2], training:False})
    print(reto)
    
   
    '''
    x_axi = np.arange(0, 4096)
    x_axi2 = np.arange(0, 3)
    y_axi = tem[0][3]
    y_axi2 = tem[1]
    fig = plt.figure()
    plt.plot(x_axi, y_axi, 'b,', label='Scatter of a+b')
    plt.xlabel('Vector Index')
    plt.ylabel('a+b')
    fig.savefig('figure_apb.png')
    plt.clf()
    plt.plot(x_axi2, y_axi2, 'b,', label='Scatter of lmod')
    plt.xlabel('Vector Index')
    plt.ylabel('lmod')
    fig.savefig('figure_lmod.png')
    print(y_axi2)
    '''
    print("Completed in %f hours" % ((t_end - t_start)/3600.))
    print("Best Accuarcy: %f, Train Epoch on which achieved best accuracy: %d" % (old_acc, store_epoch))

if __name__ == '__main__':
    main()
