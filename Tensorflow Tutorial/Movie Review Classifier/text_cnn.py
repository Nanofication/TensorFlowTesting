import tensorflow as tf
import numpy as np

class TextCNN(object):
    """
    A Convolution Neural Network for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling
    and softmax layer
    """
    def __init__(self, sequence_length, num_classes, vocab_size,
                 embedding_size, filter_sizes, num_filters):
        """
        :param sequence_length: The length of our sentences. All padded sentences have same length
        remember we need this for the input.
        :param num_classes: Number of classes in the output layer. Two in this case (Positive and negative)
        Remember: We will have about 20 or 21 depending on the number of buckets we have.
        :param vocab_size: The size of our vocabulary. This defines the size of our embedding layer
        The shape is [vocabulary_size, embedding_size]
        :param embedding_size: The dimensionality of our embeddings
        :param filter_sizes: The number of words we want our convolutional filters to cover
        For example, [3,4,5] means we have filters that slide over 3, 4, 5 words respectively.
        Total number of filters is 3.
        :param num_filters: The number of filters per filter size
        """

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prop")
        """
        tf.placeholder creates a placeholder variable that we feed to the network when
        we execute it at training or testing time. The second argument is the shape of the input tensor
        None means that the length of that dimension could be anything.
        - The first dimension is the batch size
        - None allows the network to handle arbitrarily sized batches
        - Keeping neurons in dropout layer is for training
        """

        # Embedding Layer
        """
        The first layer we define is embedding layer. This maps vocab word indices
        "indices = multiple index" into low dimensional vector representations.
        - We are essentially building a lookup table that we learn from data
        """

        with tf.device('/gpu:0'), tf.name_scope("embedding"):
            # Force GPU execution. The "embedding" scope adds all operations into a top level node
            # called "embedding" so that you get a nice hierarchy when visualizing your network in TensorBoard
            # W is our embedding matrix that we learn during training
            # initially we initialize using random uniform distribution
            W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name = "W")
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x) # Create embedding operation
            # Result is 3-dimensional tensor of shape [None, sequence_length, embedding_size]
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            # TensorFlow's convolutional conv2d operation expects 4-dimensional tensor
            # Dimensions correspond to batch, width, height and channel.



            # Convolution and Max Pooling Layers
            pooled_outputs = []
            # Each convolution produces tensors of different shapes
            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, embedding_size, 1, num_filters]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name = "W")
                    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")

                    conv = tf.nn.conv2d(
                        self.embedded_chars_expanded,
                        W,
                        strides=[1,1,1,1],
                        padding="VALID",
                        name="conv"
                    )
                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    # Max-pooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, sequence_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool"
                    )
                    pooled_outputs.append(pooled)
            # combine all the pooled features
            num_filters_total = num_filters * len(filter_sizes)
            self.h_pool = tf.concat(3, pooled_outputs)
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])