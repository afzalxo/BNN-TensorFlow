import tensorflow as tf
import numpy as np


#Implements y = +1 for x > 0 and y = -1 for x <= 0
def binarize(inp):
    h_sig = tf.clip_by_value((inp)/1., 0, 1)
    return (2.*(tf.round(h_sig)) - 1.)
    #with tf.get_default_graph().gradient_override_map({'sign': 'identitiy'}):
    #    return tf.sign(inp)

def main():
    bin_inp = np.array([-8, 0, 2, -9])
    bin_pl = tf.placeholder('float',[4])
    bin_out = binarize(bin_pl)
    f_dict = {bin_pl: bin_inp}
    with tf.device('/gpu:0'):
        with tf.Session() as sess:
            res = sess.run(bin_out, feed_dict=f_dict)
            print(res)



if __name__ == '__main__':
    main()
