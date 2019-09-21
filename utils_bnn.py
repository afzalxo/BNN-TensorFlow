import os
import tensorflow as tf
import numpy as np
from progressbar import ProgressBar
from tensorflow.keras.datasets import mnist
import time
import math

#Implements y = +1 for x > 0 and y = -1 for x <= 0
tf.logging.set_verbosity(tf.logging.ERROR)
def sign_binarize(inp):
#    h_sig = tf.clip_by_value((inp)/1., 0, 1)
#    round_out = tf.round(h_sig)
#    round_fin = h_sig + tf.stop_gradient(round_out - h_sig)
#    return (2.*round_fin - 1.)
    with tf.get_default_graph().gradient_override_map({'Sign': 'Identity'}):
        return tf.sign(tf.sign(inp)+1e-8)

def bin_dense_layer(in_act, num_out, use_bias=True, bin_inp=True, training=True, name='Bin_Dense_L'):
    with tf.variable_scope(name+'_w', reuse=False):
        cont_weights = tf.get_variable('weight', [in_act.get_shape().as_list()[1], num_out], initializer=tf.contrib.layers.xavier_initializer(), trainable=training)
    clip_w = tf.clip_by_value(cont_weights, -1, 1)
    bin_w = sign_binarize(clip_w)
    bin_act = sign_binarize(in_act) if bin_inp else in_act
    #print(name+str(bin_act.shape)+' '+str(bin_w.shape))
    res = tf.matmul(bin_act, bin_w)
    with tf.variable_scope(name+'_b', reuse=False):
        res = tf.nn.bias_add(res, tf.get_variable('bias', [num_out], initializer=tf.zeros_initializer(), trainable=training)) if use_bias else res
    #tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, res)
    return res
    
def bin_mlp_mnist(inp, use_bias=True, training=True):
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


def main():
    #Host routines credits to https://github.com/ivanbergonzani/binarized-neural-network
    timestamp = int(time.time())
    model_name = ''.join([str(timestamp), '_', 'dense', '_', 'mnist'])
    session_logdir = os.path.join('./logs/', model_name)
    train_logdir = os.path.join(session_logdir, 'train')
    test_logdir = os.path.join(session_logdir, 'test')
    session_modeldir = os.path.join('./models/', model_name)

    if not os.path.exists(session_modeldir):
        os.makedirs(session_modeldir)
    if not os.path.exists(train_logdir):
        os.makedirs(train_logdir)
    if not os.path.exists(test_logdir):
        os.makedirs(test_logdir)

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    #print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    N = tf.placeholder(tf.int64)
    data_features, data_labels = tf.placeholder(tf.float32, (None,)+x_train.shape[1:]), tf.placeholder(tf.int32, (None,)+y_train.shape[1:])

    train_data = tf.data.Dataset.from_tensor_slices((data_features, data_labels))
    train_data = train_data.repeat().shuffle(x_train.shape[0]).batch(N)

    test_data = tf.data.Dataset.from_tensor_slices((data_features, data_labels))
    test_data = test_data.repeat().shuffle(x_test.shape[0]).batch(N)

    data_iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
    
    features, labels = data_iterator.get_next()
    train_initialization = data_iterator.make_initializer(train_data)
    test_initialization = data_iterator.make_initializer(test_data)

    is_training = tf.get_variable('is_training', initializer=tf.constant(False, tf.bool))
    switch_training_inference = tf.assign(is_training, tf.logical_not(is_training))

    logits = bin_mlp_mnist(features, use_bias=True, training=is_training)

    with tf.name_scope('trainer_optimizer'):
        learning_rate = tf.Variable(1e-3, name='learning_rate')
        learning_rate_decay = tf.placeholder(tf.float32, shape=(), name='lr_decay')
        update_learning_rate = tf.assign(learning_rate, learning_rate / learning_rate_decay)
        optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
        cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        loss = tf.reduce_mean(cross_entropy)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            global_step = tf.train.get_or_create_global_step()
            train_op = optimizer.minimize(loss=loss, global_step=global_step)


    with tf.variable_scope('metrics'):
        mloss, mloss_update   = tf.metrics.mean(cross_entropy)
        accuracy, acc_update  = tf.metrics.accuracy(labels, tf.argmax(logits, axis=1))
        metrics = [mloss, accuracy]
        metrics_update = [mloss_update, acc_update]

    metrics_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
    metrics_initializer = tf.variables_initializer(metrics_variables)


# summaries
    los_sum = tf.summary.scalar('loss', mloss)
    acc_sum = tf.summary.scalar('accuracy', accuracy)
    merged_summary = tf.summary.merge([los_sum, acc_sum])


# network weights saver
    saver = tf.train.Saver()

    NUM_BATCHES_TRAIN = math.ceil(x_train.shape[0] / 32)
    NUM_BATCHES_TEST = math.ceil(x_test.shape[0] / 32)


    with tf.Session() as sess:
	# tensorboard summary writer
        train_writer = tf.summary.FileWriter(train_logdir, sess.graph)
        test_writer = tf.summary.FileWriter(test_logdir)
        sess.run(tf.global_variables_initializer())
        for epoch in range(10):
            print("\nEPOCH %d/%d" % (epoch+1, 10))
            # exponential learning rate decay
            if (epoch + 1) % 10 == 0:
                sess.run(update_learning_rate, feed_dict={learning_rate_decay: 2.0})
		
		
		# initialize training dataset and set batch normalization training
            sess.run(train_initialization, feed_dict={data_features:x_train, data_labels:y_train, N:32})
            sess.run(metrics_initializer)
            sess.run(switch_training_inference)
	    
            progress_info = ProgressBar(total=NUM_BATCHES_TRAIN, prefix=' train', show=True)
		
		# Training of the network
            for nb in range(NUM_BATCHES_TRAIN):
                sess.run(train_op)	# train network on a single batch
                batch_trn_loss, _ = sess.run(metrics_update)
                trn_loss, a = sess.run(metrics)
			
                progress_info.update_and_show( suffix = '  loss {:.4f},  acc: {:.3f}'.format(trn_loss, a) )
            print()
		
            summary = sess.run(merged_summary)
            train_writer.add_summary(summary, epoch)
		
		
		
		# initialize the test dataset and set batc normalization inference
            sess.run(test_initialization, feed_dict={data_features:x_test, data_labels:y_test, N:32})
            sess.run(metrics_initializer)
            sess.run(switch_training_inference)
		
            progress_info = ProgressBar(total=NUM_BATCHES_TEST, prefix='  eval', show=True)
		
		# evaluation of the network
            for nb in range(NUM_BATCHES_TEST):
                sess.run([loss, metrics_update])
                val_loss, a = sess.run(metrics)
			
                progress_info.update_and_show( suffix = '  loss {:.4f},  acc: {:.3f}'.format(val_loss, a) )
            print()
		
            summary  = sess.run(merged_summary)
            test_writer.add_summary(summary, epoch)
		
	
        train_writer.close()
        test_writer.close()
	
        saver.save(sess, os.path.join(session_modeldir, 'model.ckpt'))

    print('\nTraining completed!\nNetwork model is saved in  {}\nTraining logs are saved in {}'.format(session_modeldir, session_logdir))


if __name__ == '__main__':
    main()
