from bnn_misc import bin_dense_layer
from bnn_misc import dense_layer
import tensorflow as tf
from bnn_misc import sign_binarize


def binn_mlp_mnist(inp, training=True):
    alpha = .9
    epsilon = 1e-4
    inp = tf.nn.dropout(inp, rate=.2)
    inp = tf.contrib.layers.flatten(inp)
    inp = sign_binarize(inp)
    l0= bin_dense_layer(inp, 4096, training=training, name='bin_dense_l0')
    l0_bn = tf.layers.batch_normalization(l0, training=training, momentum=alpha, epsilon=epsilon)
    l0_htanh = sign_binarize(l0_bn)
    l0_htanh = tf.layers.dropout(l0_htanh, rate=.5, training=training)

    l1= bin_dense_layer(l0_htanh, 4096, training=training, name='bin_dense_l1')
    l1_bn = tf.layers.batch_normalization(l1, training=training, momentum=alpha, epsilon=epsilon)
    l1_htanh = sign_binarize(l1_bn)
    l1_htanh = tf.layers.dropout(l1_htanh, rate=.5, training=training)

    l2= bin_dense_layer(l1_htanh, 4096, training=training, name='bin_dense_l2')
    l2_bn = tf.layers.batch_normalization(l2, training=training, momentum=alpha, epsilon=epsilon)
    l2_htanh = sign_binarize(l2_bn)
    l2_htanh = tf.layers.dropout(l2_htanh, rate=.5, training=training)

    l3= bin_dense_layer(l2_htanh, 10, training=training, name='bin_dense_l3')
    l3_bn = tf.layers.batch_normalization(l3, training=training, momentum=alpha, epsilon=epsilon)

    return l3_bn


def mlp_mnist(inp, training=True):
    alpha = .9
    epsilon = 1e-4
    inp = tf.nn.dropout(inp, rate=.2)
    inp = tf.contrib.layers.flatten(inp)
    l0= dense_layer(inp, 4096, training=training, name='dense_l0')
    l0_bn = tf.layers.batch_normalization(l0, training=training, momentum=alpha, epsilon=epsilon)
    l0_htanh = l0_bn#tf.nn.relu(l0_bn)

    l1= dense_layer(l0_htanh, 4096, training=training, name='dense_l1')
    l1_bn = tf.layers.batch_normalization(l1, training=training, momentum=alpha, epsilon=epsilon)
    l1_htanh = l1_bn#tf.nn.relu(l1_bn)

    l2= dense_layer(l1_htanh, 4096, training=training, name='dense_l2')
    l2_bn = tf.layers.batch_normalization(l2, training=training, momentum=alpha, epsilon=epsilon)
    l2_htanh = l2_bn#tf.nn.relu(l2_bn)

    l3= dense_layer(l2_htanh, 10, training=training, name='dense_l3')
    l3_bn = tf.layers.batch_normalization(l3, training=training, momentum=alpha, epsilon=epsilon)
    #l3_bn = tf.nn.relu(l3_bn)
    #l3_bn = tf.nn.softmax(l3_bn)

    return l3_bn
