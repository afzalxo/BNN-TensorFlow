import tensorflow as tf
import numpy as np
#Implements y = +1 for x > 0 and y = -1 for x <= 0
#New branch
layer_names = []
def sign_binarize(inp):
    h_sig = tf.clip_by_value((inp+1.)/2., 0, 1)
    round_out = tf.round(h_sig)
    round_fin =  h_sig + tf.stop_gradient(round_out - h_sig)
    return (2.*round_fin - 1.)

def bin_dense_layer(in_act, num_out, use_bias=True, bin_inp=True, training=True, name='Bin_Dense_L'):
    with tf.variable_scope(name+'_params', reuse=False):
        l_w = tf.get_variable('weight', [in_act.shape[1], num_out], initializer = tf.random_uniform_initializer(-1, 1), constraint=lambda w: tf.clip_by_value(w, -1., 1.), trainable=True)
        l_b = tf.get_variable('bias', [num_out], initializer=tf.zeros_initializer(), trainable=True)
    tf.add_to_collection(name+'_w', l_w)
    layer_names.append(name)
    bin_w = sign_binarize(l_w)
    res = tf.matmul(in_act, bin_w)
    res = tf.nn.bias_add(res, l_b) if use_bias else res
    return res

def compute_gradients(loss, optimizer):
    gradient_list = []
    weight_updates = []
    for l_name in layer_names:
        params = tf.get_collection(l_name + '_w')
        if params:
            grad = optimizer.compute_gradients(loss, params[0])
            gradient_list.append(grad[0][0])
            weight_updates.extend(params)
    return zip(gradient_list, weight_updates)

'''
inp = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],[-2, 0, -8, -9, 3, 9, -2, 9, 0.25, 0.75, -0.25, -0.75]], dtype=np.float32)
inp1 = tf.placeholder(dtype=tf.float32, shape=(2, 12))
res = bin_dense_layer(inp1, 2048)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    ret=sess.run(res, feed_dict={inp1:inp})
    print(ret)
'''
