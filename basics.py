import tensorflow as tf

# Create constants
node1 = tf.constant(3.0)
node2 = tf.constant(4.0)

sess = tf.Session()

print(sess.run([node1, node2]))

# Add constants
node3 = tf.add(node1, node2)

print('node3: ', node3)
print('sess: ', sess.run(node3))

# Create Placeholders and add them
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adderNode = tf.add(a, b)

print(sess.run(adderNode, {a:1, b:2}))
print(sess.run(adderNode, {a:[2,3], b:[4,7]}))

# Create a node that multiplies adder with 3
tripleNode = tf.multiply(adderNode, 3)

# Run session
print(sess.run(tripleNode, {a:10, b:15}))

# Create a Linear Model y = W * x + b
W = tf.Variable([0.3], dtype=tf.float32)
b = tf.Variable([0.3], dtype=tf.float32)
x = tf.placeholder(dtype=tf.float32)

linearModel = tf.add(tf.multiply(W, x), b)

# Initialize global variales
init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(linearModel, feed_dict={x:[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}))

# Create loss function reduce_sum(square(f(x) - y))
y = tf.placeholder(tf.float32)
sq_delta = tf.square(tf.subtract(linearModel, y))
loss = tf.reduce_sum(sq_delta)

print(sess.run(loss, feed_dict={x: [0, 1, 2, 3], y: [1, 2, 3, 4]}))

# Assign new values to variables
fixW = tf.assign(W, [1])
fixb = tf.assign(b, [1])

# Run with corrected weight
sess.run([fixb, fixW])
print(sess.run(loss,feed_dict={x: [0, 1, 2, 3], y: [1, 2, 3, 4]}))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(loss)


x_train = [0, 1, 2, 3, 4]
y_train = [1, 2, 3, 4, 5]
# Back to wrong values
sess.run(init)
for i in range(1000):
    sess.run(train, feed_dict={x:x_train, y:y_train})
    print(sess.run([W, b]))
