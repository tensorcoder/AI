import tensorflow as tf
import numpy as np
from ImageAI.imageimport import

"""
THIS IS THE RESHAPE FUNCTION
a = tf.truncated_normal([16,128,128,3])
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(tf.shape(a)))

b = tf.reshape(a, [16, 49152])
print(sess.run(tf.shape(b)))
"""

"""
THIS IS THE SOFTMAX FUNCTION
A function which converts K-dimensional vector "X" containing real values to the same shaped vector of real values in 
range (0,1) whose sum is 1. We shall apply the softmax function to the output of our convolutional neural networks in 
order to convert the output to the probability for each class. 
$o(x)_j = (exp(x_i)/sum(n=1 to N) exp(x_n)) for j=1 to N
"""

"""
I'm going to use 500 images of cats and dogs each. We divide this "training data" into sections: 
1) Training data: 80% ie 800 images
2) Validation data: 20% images taken out of training data to calculate accuracy independently during the training process
3) Test set: separate and independent data for testing. 

Sometimes due to something called Overfitting, after training, neural networks start working very well on the training 
data and very similar images. But they fail to work well for other images. For example, if you are training a classifier
between dogs and cats and you get training data from someone who takes all images with white backgrounds, it's possible 
that your network works very well on this validation data set but if you try to run it on an image with a cluttered 
background, it will most likely fail. So that's why we try to get our test-set from an independent source. 
"""

classes = ['dogs', 'cats']
num_classes = len(classes)

train_path = 'training_data'
test_path = 'testing_data'

#validation split
validation_size = 0.2

#batch size
batch_size = 16

data = DataSet.read_train_sets(train_path, img_size, classes, validation_size=validation_size)
test_images, test_ids = DataSet.read_test_set(test_path, img_size)


def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))


"""
Building convolution layer in TensorFlow:
tf.nn.conv2d(input, filter, strides, padding='SAME')  <-- can be used to build a convolutional layer

    input = the output(activation) from the previous layer. This should be a 4D tensor.
    Typically, in the first convolutional layer, you pass n images of size: 
    width*height*num_channels <-- then this has the size [n width height num_channels]

    filter = trainable variables defining the filter. We start with a random normal distribution
    and learn these weights. It's a 4D tensor whose specific shape is predefined as part of network 
    design. If your filter is of size filter_size and input feed has num_input_channels and you have 
    num_filters filter in your current layer, then filter will have the following shape:
        [filter_size filter_size num_input_channels num_filters]

    strides = defines how much you move your filter when doing convolution. In this function, it needs 
    to be a Tensor of size >= 4 ie [batch_stride x_stride y_stride depth_stride] 
    batch_stride is always 1 as you don't want to skip images in your batch. 
    x_stride and y_stride are same mostly and the choice is part of network design and we shall use them as 1
    depth_stride is always set to 1 as you don't skip along the depth. 


    padding=SAME means we shall pad the input image with 0s in such a way that output x,y dimensions are 
    the same as that of input
"""

"""
After convolution in Tensorflow:
    After convolution we add the biases of that neuron, which are also learnable/trainable. Again we start
    with random normal distribution and learn these values during training. 
"""

"""
Max Pooling: 

    tf.nn.max_pool(value=layer, 
                                ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1], 
                                padding='SAME')

"""
# Here we are defining a function to create a complete convolutional layer
def new_conv_layer(input, num_input_channels, filter_size, num_filters, use_pooling=True):

# The output of this layer is a multi-dimensional Tensor.
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    weights = new_weights(shape=shape)

    biases = new_biases(length=num_filters)

    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    layer += biases

    if use_pooling:
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2 , 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    layer = tf.nn.relu(layer)

    return layer, weights


"""
Since the output of the convolutional layer is a multi-dimensional Tensor, we need to convert it into a one-D tensor. 
This is done in the flatting layer. We use the reshape operation to create a single dimensional tensor. 
"""


def flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()

    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features

"""
Now we have to define a function to create a fully connected layer. Just like any other layer, we declare weights and 
biases as random normal distributions. In fully connected layer, we take all the inputs and do the standard z=wx+b 
operation on them. Also sometimes you would want to add a non-linearity(RELU) to it. So, let's add a condition that 
allows the caller to add RELU to the layer. 
"""


def new_fc_layer(input,
                 num_inputs,
                 num_outputs,
                 use_relu=True):

    weights =     new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    layer = tf.add(tf.matmul(input, weights), biases)
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


"""
Placeholders and input: 

    Now that we've finished defining all the building blocks of the network, let's create a placeholder that will hold the
    input training images. All the input images are read in imageimport.py file and resized to 128x128x3. While reading
    images, instead of reading them in a multi-dimensional Tensor, we read them into a single dimensional Tensor of size
    49152 (128*128*3) for simplicity. 

    The input placeholder x is created in the shape of [None, 49152]. The first dimension= None means you can pass any 
    number of images to it. 

    We shall pass images in the batch of 16, ie shape will be [16, 49152]. After this, we reshape this into [16 128 128 3]
    Similarly, we create a placeholder y_true for storing the predictions. For each image, we have two outputs(probabilites)
    for each class --> y_pred is of shape [None 2] (for a batch size of 16 it will be [16 2]
"""

img_size_flat = img_size * img_size * num_channels

x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x') #name your placeholders so tf doesn't name 4 U
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])

y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1) # what does dimension = 1 do???


"""
Network Design: 
    
    We use the functions defined above to create various layers of the network. 

"""
layer_conv1, weights_conv1 = \
    new_conv_layer(input=x_image, num_input_channels=num_channels, filter_size=filter_size1, num_filters=num_filters1,\
                   use_pooling=True)

layer_conv2, weights_conv2 = \
new_conv_layer(input=layer_conv1, num_input_channels=num_filters1, filter_size=filter_size2, num_filters=num_filters2,\
               use_pooling=True)

layer_conv3, weights_conv3 = \
new_conv_layer(input=layer_conv2, num_input_channels=num_filters2, filter_size=filter_size3, num_filters=num_filters3,\
               use_pooling=True)

layer_flat, num_features = flatten_layer(layer_conv3)

layer_fc1 = new_fc_layer(input=layer_flat, num_inputs=num_features, num_outputs=fc_size, use_relu=True)

layer_fc2 = new_fc_layer(input=layer_fc1, num_inputs=fc_size, num_outputs=num_classes, use_relu=False)

"""
Predictions: 
    You can get the probability of each class by applying softmax to the output of the fully connected layer.
    
    y_pred = tf.nn.softmax(layer_fc2)
    
    y_pred contains the predicted probability of each class for each input image. the class having higher
    probability is the prediction of the netowrk. 
    
    y_pred_cls = tf.argmax(y_pred, dimension=1) 
    
"""

y_pred = tf.nn.softmax(layer_fc2)
y_pred_cls = tf.argmax(y_pred, dimension=1)

"""
Cost: 
    Now let's define the cost that will be minimized to reach the optimum value of weights. We will use a simple cost
    that will be calculated using a Tensorflow function softmax_cross_entropy_with_logits which takes the output of the
    last fully connected layer and actual labels to calculated cross_entropy whose average will give us the cost.
"""

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_true)
cost = tf.reduce_mean(cross_entropy)


"""
Optimization: 
    Tensorflow implements most of the optimisation functions. We shall use AdamOptimizer for gradient calculation and 
    weight optimization. We shall specify that we are trying to minimise cost with a learning rate of 0.0001. """


optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

"""
If we run optimizer inside session.run(), in order to calculate the value of cost, the whole networks will have to be 
run and we will pass the training images in a feed_dict. Training images are passed in a batch of 16(train_batch_size)
in each iteration. 
"""

train_batch_size = batch_size

x_batch, y_true_batch, _, cls_batch = data.train.next_batch(train_batch_size)
x_batch = x_batch.reshape(train_batch_size, img_size_flat)

feed_dict_train = {x: x_batch, y_true: y_true_batch}

session.run(optimizer, feed_dict=feed_dict_train)


x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(train_batch_size)

x_valid_batch = x_valid_batch.reshape(train_batch_size, img_size_flat)

feed_dict_validate = {x: x_valid_batch,
                      y_true: y_valid_batch}

val_loss = session.run(cost, feed_dict=feed_dict_validate) # passing cost in session with validation images instead of\
#training images

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

"""
We can calculate validation accuracy by passing accuracy in session.run() and providing validation images in a feed_dict
"""

val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
acc = session.run(accuracy, feed_dict=feed_dict_train)

"""
As training images along with labels are used for training, so training accuracy will be higher than validation. 
We report training accuracy to know that we are at least moving in the right direction and are at least improving
accuracy in the training dataset.
"""

total_iterations = 0

def optimize(num_iterations):
    global total_iterations

    best_val_loss = float("inf")
    patience = 0

    for i in range(total_iterations, total_iterations + num_iterations):

        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(train_batch_size)

        x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(train_batch_size)

        x_batch = x_batch.reshape(train_batch_size, img_size_flat)
        x_valid_batch = x_valid_batch.reshape(train_batch_size, img_size_flat)
        feed_dict_train ={x: x_batch, y_true: y_true_batch}

        feed_dict_validate = {x: x_valid_batch, y_true: y_valid_batch}

        session.run(optimizer, feed_dict=feed_dict_train)

        if i % int(data.train.num_examples/batch_size) == 0:
            val_loss = session.run(cost, feed_dict=feed_dict_validate)
            epoch = int(i / int(data.train.num_examples/batch_size))

            print_progress(epoch, feed_dict_train, feed_dict_validate, val_loss)
    total_iterations += num_iterations