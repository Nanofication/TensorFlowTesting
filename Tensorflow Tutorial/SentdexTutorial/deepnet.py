import tensorflow as tf

"""
1. Input Data --> Weight > Hidden Layer 1 (activation function) --> weights > Hidden Layer 2
(activation function) > weights > output layer (FEED FORWARD NN)

2. Compare output to intended output > cost or loss function (cross entropy How wrong are we?)
Optimization function (optimizer) > minimize cost (AdamOptomizer .. SGD, AdaGrad, etc)

3. Backpropogation

4. Feed Forward + Backpropogation = Epoch (1 cycle)
"""

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data", one_hot= True)
# one_hot means one component will be hot and rest are off.

# Output from our neural network will have ~10 output nodes

# 10 Classes, 0 - 9 hand written digits
"""

0 = 0
1 = 1
2 = 2

one_hot does --> 0 = [1,0,0,0,0,0,0,0,0,0]
one_hot does --> 1 = [0,1,0,0,0,0,0,0,0,0]
one_hot does --> 2 = [0,0,1,0,0,0,0,0,0,0]
one_hot does --> 3 = [0,0,0,1,0,0,0,0,0,0]

"""

# These are hidden layers, just do it and don't mind.
# Tweak it to your hearts content
n_nodes_hl1 = 500 # just say this
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100 # Go through batches of 100 features and feeds them into network 1 at a time

x = tf.placeholder('float', [None, 784]) # This is your input data In this case its 784 pixels. (Squashed 28 x 28)
y = tf.placeholder('float') # This is your output data

def neural_network_model(data):
    hidden_1_layer = {'weights':tf.Variable(
        tf.random_normal([
            784, n_nodes_hl1
        ])), # All of your weights. Create an array of your data using a bunch of random numbers and that's your weights
        'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))
    }
    # Biases something added after the weights. Input data is multiplied by the weights.
    # Biases adds to the entire thing
    # Formula: input_data * weights + biases

    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
        'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
        'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
        'biases': tf.Variable(tf.random_normal([n_classes]))}

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1) # Activation function

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2) # Activation function

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3) # Activation function

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return output

def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

    # Has learning rate = 0.001
    optimizer = tf.train.AdamOptimizer().minimize(cost) #Like stochastic gradient descent

    # Cycles feed forward + back prop
    hm_epochs = 10

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        # This is for training the data
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)): # _ means variable we don't care about
                epoch_x, epoch_y = mnist.train.next_batch(batch_size) # This chunks through the dataset for you
                # We need to build this for ourselves in the future
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y}) # c = cost
                epoch_loss += c
            print('Epoch: ', epoch, ' completed out of ', hm_epochs, ' loss: ', epoch_loss)

        # We run them through our model
        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1)) # Return the index of the maximum value. Hoping the index values are the same
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy: ', accuracy.eval({x: mnist.test.images, y:mnist.test.labels})) # evaluate all images of test images with test labels

train_neural_network(x)