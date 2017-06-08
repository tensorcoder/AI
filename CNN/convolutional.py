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

#10 classes, 0 - 9 handwritten digits

"""
This is what one_hot = True means : 
Since we have 0-9 handwritten digits, only one of them will be in the picture at a time, which means one_hot =True
0 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] 
1 = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
2 = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0] 
"""
#here we have 3 hidden layers of neurons with 500 nodes each


#numbers of classes
n_classes = 10

#you need to define a batch size to load into your RAM because you can't load the entire dataset at once
batch_size = 128

#define placeholder variables height x width
x = tf.placeholder(tf.float32, shape=[None, 784]) # with the defined size if something goes wrong tensorflow will throw an error but without it it won't
y = tf.placeholder(tf.float32, shape=[None, 10])

keep_rate = 0.8 # for dropout
keep_prob = tf.placeholder(tf.float32) # for dropout
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def convolutional_neural_network(x):
    weights = {'W_conv1': tf.Variable(tf.random_normal([5, 5 ,1, 32])),  #5x5conv, 1 input, produces 32 features
               'W_conv2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
               'W_fc': tf.Variable(tf.random_normal([7*7*64, 1024])),
               'out': tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1': tf.Variable(tf.random_normal([32])),  #5x5conv, 1 input, produces 32 features
               'b_conv2': tf.Variable(tf.random_normal([64])),
               'b_fc': tf.Variable(tf.random_normal([1024])),
               'out': tf.Variable(tf.random_normal([n_classes]))}

    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])

    conv1 = maxpool2d(conv1)

    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2, [-1, 7*7*64])

    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])

    fc = tf.nn.dropout(fc, keep_rate) # for dropout

    output = tf.matmul(fc, weights['out']) + biases['out']

    return output


def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y)) #your output will always be the shape of your testing set's labels
    #                           learning_rate = 0.001 learning rate by default
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 5 # cycles of feed forward +backpropagation aka the number of times you feed the information through your neural network

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs): # this section here is just training the data in some way
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)): # this = total num of samples / batch size = how many times we need to cycle
                epoch_x, epoch_y = mnist.train.next_batch(batch_size) # chunks through the data set for you
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            print('Epoch', epoch+1, 'completed out of', hm_epochs, 'loss: ', epoch_loss)

            #once we have trained our ai, we test it here.
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1)) #tf.argmax returns the index of the maximum value, this code comparing the prediction to the answer
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy: ', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

train_neural_network(x)

graph = tf.get_default_graph()
graph.get_operations()


'''
There is such a thing as dropout, which is popular with convolutional neural nets because it helps to resolve local max
ima. It does this by killing some neurons which would otherwise lead to overfitting

This is the best type of network for image data usually but the RNN is better for smaller sets. 

'''




















"""
biases are things that are added at the end after the weights
the weights are multiplied by the weights and the weight 

input_data * weights + biases 

the biggest benefit of biases is that if all the input data was 0 the weights times 0 would be 0 so the neurons won't fire
with a bias however, it will fire because something is added to those zeros 

"""



