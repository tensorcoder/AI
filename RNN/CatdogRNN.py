import tensorflow as tf
import os
import glob
import numpy as np
from tensorflow.contrib import rnn
from RNN.AnyImage import *



""" The Import from .AnyImage contains: 
1) "classes" definition
2) "image_size" definition
3) "test_path" definition 
4) "train_path" definition
5) load_train, load_test, class:DataSet(), read_train_sets, read_test_set 
"""

""" 
RNN Architecture 
input > weights > hidden layer 1 (activation function) > weights > hidden layer2 
(activation function) > weights > output layer
"""


#Parameters of the RNN defined here.
hm_epochs = 5   # Number of cycles of feedforward + backprop. AKA how many times all images pass through NN Higher will improve accuracy
n_classes = 2    # Number of classes to learn.
batch_size = 10 # How many images we want to load up at a time.
chunk_size = 128 # with RNN we need to go in a certain order. The images are 128*128 pix so we will go in 128 pix chunks
n_chunks = 128    # 128 times. # This is the "LSTM" memory we are talking about
rnn_size = 512   # can make this bigger

#defining the data set to be anything you want: not only mnist
data_sets = read_train_sets(train_path, image_size, classes, validation_size)
data_sets_test_images, data_sets_test_labels = read_test_set(test_path, image_size, classes)

#define placeholder variables height x width
x = tf.placeholder(dtype=tf.float32, shape=[None, n_chunks, chunk_size]) # with the defined size if something goes wrong tensorflow will throw an error but without it it won't
y = tf.placeholder(dtype=tf.float32, shape=[None, n_classes])

def recurrent_neural_network(x):
    layer = {'weights': tf.Variable(tf.random_normal([rnn_size, n_classes])),
                      'biases': tf.Variable(tf.random_normal([n_classes]))} #biases are something that is added after

    x = tf.transpose(x, [1, 0, 2])  #formatting data so tensorflow is content
    x = tf.reshape(x, [-1, chunk_size])
    x = tf.split(x, n_chunks, 0)

    lstm_cell = rnn.BasicLSTMCell(rnn_size)

    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']

    return output


def train_neural_network(x):
    prediction = recurrent_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y)) #your output will always be the shape of your testing set's labels
    #                           learning_rate = 0.001 learning rate by default
    optimizer = tf.train.AdamOptimizer().minimize(cost)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs): # this section here is just training the data in some way
            epoch_loss = 0
            for _ in range(int(data_sets.train._num_examples/batch_size)): # this = total num of samples / batch size = how many times we need to cycle
                epoch_x, epoch_y, _v_, cls_batch = data_sets.train.next_batch(batch_size) # chunks through the data set for you
                epoch_x = epoch_x.reshape((batch_size, n_chunks, chunk_size))
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            print('Epoch', epoch+1, 'completed out of', hm_epochs, 'loss: ', epoch_loss)

            #once we have trained our ai, we test it here.
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1)) #tf.argmax returns the index of the maximum value, this code comparing the prediction to the answer
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        # print('test images: ', data_sets_test_images, 'test_labels: ', data_sets_test_labels)
        print('Accuracy: ', accuracy.eval({x: data_sets_test_images.reshape((-1, n_chunks, chunk_size)), y: data_sets_test_labels}))

train_neural_network(x)

graph = tf.get_default_graph()
graph.get_operations()























"""
biases are things that are added at the end after the weights
the weights are multiplied by the weights and the weight 

input_data * weights + biases 

the biggest benefit of biases is that if all the input data was 0 the weights times 0 would be 0 so the neurons won't fire
with a bias however, it will fire because something is added to those zeros 

"""