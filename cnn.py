import argparse
import sys
import tempfile

from classification import next_batch
from CnnData import CnnData

import tensorflow as tf
import numpy as np

FLAGS = None
num_class = 14
image_width = 28
image_height = 28
image_depth = 1
epochs = 20000
cv_fold = 5
batchSize = 256

cnnData = CnnData()


def deepnn(x):
    """deepnn builds the graph for a deep net for classifying Chinese characters.
    Args:
      x: an input tensor with the dimensions (N_examples, 784), where 784 is the
      number of pixels in a image.
    Returns:
      A tuple (y, keep_prob). y is a tensor of shape (N_examples, 14), with values
      equal to the logits of classifying the digit into one of 14 classes (the
      14 Chinese characters). keep_prob is a scalar placeholder for the probability of
      dropout.
    """
    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 28, 28, 1])  # -1 means unknown #samples

    # First convolutional layer - maps one grayscale image to 32 feature maps.
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([3, 3, 1, 32])  # TODO: #filters, filter size
        b_conv1 = bias_variable([32])
        conv1_out = conv2d(x_image, W_conv1) + b_conv1
        # BN
        mode = tf.placeholder(tf.bool)
        bn1 = tf.layers.batch_normalization(conv1_out, scale=False, training=mode)
        h_conv1 = tf.nn.relu(bn1)

    # Pooling layer - downsamples by 2X.
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    # Second convolutional layer -- maps 32 feature maps to 64.
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([3, 3, 32, 64])  # TODO: #filters, filter size/ depth
        b_conv2 = bias_variable([64])
        conv2_out = conv2d(h_pool1, W_conv2) + b_conv2
        bn2 = tf.layers.batch_normalization(conv2_out, scale=False, training=mode)
        h_conv2 = tf.nn.relu(bn2)


    # Second pooling layer.
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        fc1_out = tf.matmul(h_pool2_flat, W_fc1) + b_fc1
        bn3 = tf.layers.batch_normalization(fc1_out, scale=False, training=mode)
        h_fc1 = tf.nn.relu(bn3)

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Map the 1024 features to 14 classes, one for each Chinese character
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([1024, 14])
        b_fc2 = bias_variable([14])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y_conv, keep_prob, mode


def batch_norm_layer(x, train_phase, scope_bn):
    with tf.variable_scope(scope_bn):
        beta = tf.Variable(tf.constant(0.0, shape=[x.shape[-1]]), name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[x.shape[-1]]), name='gamma', trainable=True)
        axises = np.arange(len(x.shape) - 1)
        batch_mean, batch_var = tf.nn.moments(x, axises, name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(train_phase, mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')  # TODO: stride, padding


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def main(_):
    global cnnData
    for r in range(cv_fold):
        cnnData.nextCVRound()

        # Create the model
        x = tf.placeholder(tf.float32, [None, 784])

        # Define loss and optimizer
        y_ = tf.placeholder(tf.float32, [None, 14])

        # Build the graph for the deep net
        y_conv, keep_prob, mode = deepnn(x)

        with tf.name_scope('loss'):  # TODO: L2正则项
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                                    logits=y_conv)
        cross_entropy = tf.reduce_mean(cross_entropy)

        with tf.name_scope('adam_optimizer'):  # TODO: optimizer in TF, learning rate
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
            correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)

        graph_location = tempfile.mkdtemp()  # TODO
        print('Saving graph to: %s' % graph_location)
        train_writer = tf.summary.FileWriter(graph_location)
        train_writer.add_graph(tf.get_default_graph())

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(epochs):
                for (Xb, Yb) in next_batch(cnnData.X_train, cnnData.Y_train, batchSize):
                    train_step.run(feed_dict={x: Xb, y_: Yb, keep_prob: 0.5, mode: True})

                if i % 25 == 0:  # log
                    train_accuracy = accuracy.eval(feed_dict={
                        x: cnnData.X_train, y_: cnnData.Y_train, keep_prob: 1.0, mode: False})
                    print('step %d, training accuracy %g' % (i, train_accuracy))
                    if train_accuracy == 1:
                        break

            print('test accuracy %g' % accuracy.eval(feed_dict={
                x: cnnData.X_test, y_: cnnData.Y_test, keep_prob: 1.0, mode: False}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='tmp',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
