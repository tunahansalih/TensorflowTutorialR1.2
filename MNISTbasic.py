import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
plt.figure()
plt.imshow(mnist.train.images[2].reshape((28,28)))

x = tf.placeholder(tf.float32, [None, 784])

W = tf.Variable(initial_value=np.zeros([784, 10]), dtype=tf.float32)
b = tf.Variable(initial_value=np.zeros([10]), dtype=tf.float32)

y = tf.nn.softmax(tf.add(tf.matmul(x, W), b))

y_expected = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(tf.multiply(y_expected, tf.log(y)), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
        batch_x, batch_y = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_x, y_expected: batch_y})

correct_predictions = tf.equal(tf.argmax(y,1), tf.argmax(y_expected,1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_expected: mnist.test.labels}))