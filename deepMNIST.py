import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

x = tf.placeholder(tf.float32, shape=[None, 784])
y_expected = tf.placeholder(tf.float32, shape=[None, 10])

# First step reshape X to 28*28*1, -1 means that the tensor will take what width it needs
x_images = tf.reshape(x, shape=[-1, 28, 28, 1])

# First Convolution Layer
W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))

# The output will still be 28*28
h_conv1 = tf.nn.relu(tf.nn.conv2d(input=x_images,
                                  filter=W_conv1,
                                  strides=[1, 1, 1, 1],
                                  padding='SAME') + b_conv1)

# Pooling Layer 28*28 will be 14*14
h_pool1 = tf.nn.max_pool(h_conv1,
                         ksize=[1, 2, 2, 1],
                         strides=[1, 2, 2, 1],
                         padding='SAME')

# Second Convolution Layer
W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32,  64], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))

# The output will be 14*14
h_conv2 = tf.nn.relu(tf.nn.conv2d(input=h_pool1,
                                  filter=W_conv2,
                                  strides=[1, 1, 1, 1],
                                  padding='SAME') + b_conv2)

# The output will be 7*7
h_pool2 = tf.nn.max_pool(h_conv2,
                         ksize=[1, 2, 2, 1],
                         strides=[1, 2, 2, 1],
                         padding='SAME')

# Flatten the images
h_pool2_reshape = tf.reshape(h_pool2, shape=[-1, 7*7*64])

# Dense Layer
W_dl1 = tf.Variable(tf.truncated_normal(shape=[7*7*64, 1024], stddev=0.1))
b_dl1 = tf.Variable(tf.constant(0.1, shape=[1024]))

# Multiply with weights
h_dl1 = tf.nn.relu(tf.matmul(h_pool2_reshape, W_dl1) + b_dl1)

# Dropout Layer
keep_prop = tf.placeholder(tf.float32)
h_dl1_drop = tf.nn.dropout(h_dl1, keep_prop)

# Readout layer(Dense Layer 2)
W_dl2 = tf.Variable(tf.truncated_normal(shape=[1024, 10], stddev=0.1))
b_dl2 = tf.Variable(tf.constant(0.1, shape=[10]))

y_conv = tf.matmul(h_dl1_drop, W_dl2) + b_dl2

# Train and evaluate
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_expected,
                                                                       logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_expected, 1), tf.argmax(y_conv, 1)), tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
        x_batch, y_batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_acc = sess.run(accuracy, feed_dict={x: x_batch,
                                                      y_expected: y_batch,
                                                      keep_prop: 1.0})
            print("Step: %d, Accuracy: %06.5f" % (i, train_acc))
        sess.run(train_step, feed_dict={x: x_batch,
                                        y_expected: y_batch,
                                        keep_prop: 1.0})

    test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images,
                                               y_expected: mnist.test.labels,
                                               keep_prop: 1.0})
    print("Test accuracy: %06.5f" % test_acc)

#
#
#
#
#
# init = tf.global_variables_initializer()
# sess = tf.Session()
# sess.run(init)
#
# x_batch, y_batch = mnist.train.next_batch(100)
# sess.run(h_dl1, feed_dict={x: x_batch, y_expected: y_batch})
#
