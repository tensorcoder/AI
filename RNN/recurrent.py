
"""

with a traditional nn we have to concept of the order of events
a traditional nn would not know the difference between billy killed jim and jim killed billy
you have to know if the ball is moving forward or backward

"""


import tensorflow as tf

#we're working with the mnist data set of 60,000 of handwritten 28*28 pixels and 10,000 testing examples
#objective is to take the mnist handwritten characters and pass through neural network until it can model
#what is going on.

""" input > weight > hidden layer 1 (activation function) > weights > hidden layer2 
(activation function) > weights > output layer"""

# in a typical neural network we just pass data straight through (feed forward)
#at the end compare output to intended output > cost functior (loss function) cross entropy is an example
#then gonnna use (optimizer) > minimize cost (AdamOptimizer... SGD, AdaGrad) < 8 different options in tensorflow
#that goes backwards and it's called backpropagation

#feed forward + backpropagation = epoch (that's the cycle)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True) #this means one component has electrcity running through it and the rest don't

from tensorflow.contrib import rnn

#10 classes, 0 - 9 handwritten digits

"""
This is what one_hot = True means : 
Since we have 0-9 handwritten digits, only one of them will be in the picture at a time, which means one_hot =True
0 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] 
1 = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
2 = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0] 
"""
hm_epochs = 5  # cycles of feed forward +backpropagation aka the number of times you feed the information through your neural network

#numbers of classes
n_classes = 10

#you need to define a batch size to load into your RAM because you can't load the entire dataset at once
batch_size = 128
chunk_size = 28
n_chunks = 28
rnn_size = 128 # can make this bigger

#define placeholder variables height x width
x = tf.placeholder(dtype=tf.float32, shape=[None, n_chunks, chunk_size]) # with the defined size if something goes wrong tensorflow will throw an error but without it it won't
y = tf.placeholder(dtype=tf.float32, shape=[None, 10])

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
            for _ in range(int(mnist.train.num_examples/batch_size)): # this = total num of samples / batch size = how many times we need to cycle
                epoch_x, epoch_y = mnist.train.next_batch(batch_size) # chunks through the data set for you
                epoch_x = epoch_x.reshape((batch_size, n_chunks, chunk_size))
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            print('Epoch', epoch+1, 'completed out of', hm_epochs, 'loss: ', epoch_loss)

            #once we have trained our ai, we test it here.
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1)) #tf.argmax returns the index of the maximum value, this code comparing the prediction to the answer
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy: ', accuracy.eval({x: mnist.test.images.reshape((-1, n_chunks, chunk_size)), y: mnist.test.labels}))

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


