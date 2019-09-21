from binn_misc import bin_dense_layer
import tensorflow as tf

def binn_mlp_mnist(inp, use_bias=True, training=True):
    inp = tf.contrib.layers.flatten(inp)
    l0 = bin_dense_layer(inp, 2048, use_bias=use_bias, bin_inp=False, training=training, name='bin_dense_l0')
    l0_bn = tf.layers.batch_normalization(l0, training=training)
    l0_htanh = tf.clip_by_value(l0_bn, -1, 1)

    l1 = bin_dense_layer(l0_htanh, 2048, use_bias = use_bias, bin_inp=True, training=training, name='bin_dense_l1')
    l1_bn = tf.layers.batch_normalization(l1, training=training)
    l1_htanh = tf.clip_by_value(l1_bn, -1, 1)

    l2 = bin_dense_layer(l1_htanh, 2048, use_bias=use_bias, bin_inp=True, training=training, name='bin_dense_l2')
    l2_bn = tf.layers.batch_normalization(l2, training=training)
    l2_htanh = tf.clip_by_value(l2_bn, -1, 1)

    l3 = bin_dense_layer(l2_htanh, 10, use_bias=use_bias, bin_inp=True, training=training, name='bin_dense_l3')
    l3_bn = tf.layers.batch_normalization(l3, training=training)

    return l3_bn

