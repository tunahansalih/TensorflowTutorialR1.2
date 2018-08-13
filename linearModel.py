import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


x_train = np.linspace(0.0, 100.0, 101)
y_train = 0.1 * np.linspace(0.0, 100.0, 101) - 0.01

# Add some noise to y_train
y_train += np.random.normal(loc=0.0, scale=1.0, size=(101,))

# Plot x_train vs y_train
plt.figure(figsize=(10, 10))
plt.scatter(x_train, y_train)
plt.show()

W = tf.Variable(1.0)
b = tf.Variable(1.0)
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
init = tf.global_variables_initializer()

linearModel = tf.add(tf.multiply(W, x), b)
loss = tf.reduce_sum(tf.square(tf.subtract(linearModel, y)))

optimizer = tf.train.GradientDescentOptimizer(0.000001)
train = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(init)
    for i in range(100):
        sess.run(train, feed_dict={x: x_train, y: y_train})
        curr_W, curr_b, curr_loss = sess.run([W, b, loss], feed_dict={x: x_train, y: y_train})

x_curr = np.linspace(0,100,101)
y_curr = np.add(np.multiply(x_curr, curr_W), curr_b)

plt.figure()
plt.scatter(x_train, y_train)
plt.plot(x_curr, y_curr)
plt.show()



