import tensorflow as tf
import numpy as np


c = tf.Variable([10, 20, 30, 40, 50, 60], name='t')
a = np.array([[0.1, 0.2, 0.3, 3, 5, 6], [20, 2, 3, 8, 9, 12], [1, 3, 4, 90, 8, 7]])
b = tf.Variable(a, name='b')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(tf.reduce_mean(c)))
    print(sess.run(tf.argmax(b, 2)))#shows the index of the maximum value in each row of a [5 0]
