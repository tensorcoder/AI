import tensorflow as tf
#NumPy is often used to load, manipulate and reprocess data
import numpy as np

#Declare list of features. We only have one real- valued feature.
#There are many other types of columns that are more complicated and useful.

# features = [tf.contrib.layers.real_valued_column("x", dimension=1)]
#
# # An estimator is the front end to invoke training (fitting) and evaluation
# # (inference). There are many predefined types like linear regression,
# # logistic regression, linear classification, logistic classification, and
# # many neural network classifiers and regressors. The following code
# # provides an estimator that does linear regression.
# #
# estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)
# # TensorFlow provides many helper methods to read and set up data sets.
# # Here we use `numpy_input_fn`. We have to tell the function how many batches
# # of data (num_epochs) we want and how big each batch should be.
#
# x = np.array([1., 2., 3., 4.])
# y = np.array([0., -1., -2., -3.])
# input_fn = tf.contrib.learn.io.numpy_input_fn({"x": x}, y, batch_size=4,
#                                               num_epochs=1000)
#
# #We can invoke a 1000 training steps by invoking the 'fit' method and passing the
# #training data set.
#
# estimator.fit(input_fn=input_fn, steps=1000)
#
# # Here we evaluate how well our model did. In a real example, we would want
# # to use a separate validation and testing data set to avoid overfitting.
#
# print(estimator.evaluate(input_fn=input_fn))

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
y = tf.multiply(a, b)
feed_dict = {a: 2, b: 3}
with tf.Session() as sess:
    print(sess.run(y, feed_dict))
    #sess.run(tf.global_variables_initializer())

w = tf.Variable(tf.random_normal([784, 10], stddev=0.01))

c = tf.Variable([10, 20, 30, 40, 50, 60], name='t')
a = np.array([[0.1, 0.2, 0.3, 3, 5, 6], [20, 2, 3, 8, 9, 12]])
b = tf.Variable(a, name='b')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(tf.reduce_mean(c)))
    print(sess.run(tf.argmax(b,1))  #shows the index of the maximum value in each row of a [5 0]
#

# trainX = np.linspace(-1, 1, 101)
# trainY = 3 * trainX + np.random.randn(*trainX.shape) * 0.33
#
# X = tf.placeholder(tf.float32)
# Y = tf.placeholder(tf.float32)
# w = tf.Variable(0.0, name='weights')
# y_model = tf.multiply(X, w)
#
# cost = (tf.pow(Y-y_model, 2))
# train_adam = tf.train.AdamOptimizer(0.0001).minimize(cost)
# train_gradient = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)
# init = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init)
#     for i in range(1000):
#         for (x, y) in zip(trainX, trainY):
#
#             sess.run(train_gradient, feed_dict={X: x, Y: y})
#
#
#     print('gradient: ', sess.run(w))
#
# with tf.Session() as sess:
#     sess.run(init)
#     for io in range(1000):
#         for (x, y) in zip(trainX, trainY):
#             sess.run(train_adam, feed_dict={X: x, Y: y})
#     print('adam w: ', sess.run(w))