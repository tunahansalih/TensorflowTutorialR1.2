import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, shape=[None, 784])
y_expected = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(np.zeros([784, 10]), dtype=tf.float32)
b = tf.Variable(np.zeros([10]), dtype=tf.float32)

y_predicted = tf.add(tf.matmul(x, W), b)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_expected, logits=y_predicted))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
for i in range(1000):
    x_batch, y_batch = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: x_batch, y_expected: y_batch})

correct_predictions = tf.cast(tf.equal(tf.argmax(y_predicted, 1), tf.argmax(y_expected, 1)), tf.float32)
accuracy = tf.reduce_mean(correct_predictions)

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_expected: mnist.test.labels}))