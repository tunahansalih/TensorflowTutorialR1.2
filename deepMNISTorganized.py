import os
import time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

NUM_CLASSES = 10
INPUT_SIZE = 28*28
HIDDEN_UNITS = [32, 64]
BATCH_SIZE = 50
LEARNING_RATE = 0.001
MAX_STEPS = 100000

def inference(train_data):

    with tf.name_scope(name='hidden_layer_1'):
        weights = tf.Variable(tf.truncated_normal(shape=[INPUT_SIZE, HIDDEN_UNITS[0]],
                                                  stddev=0.1,),
                              name='weights')
        bias = tf.Variable(tf.zeros(shape=[HIDDEN_UNITS[0]]),
                           name='bias')
        h_0 = tf.nn.relu(tf.matmul(train_data, weights) + bias)

    with tf.name_scope(name='hidden_layer_2'):
        weights = tf.Variable(tf.truncated_normal(shape=[HIDDEN_UNITS[0], HIDDEN_UNITS[1]],
                                                  stddev=0.1),
                              name='weights')
        bias = tf.Variable(tf.zeros(shape=[HIDDEN_UNITS[1]]),
                           name='bias')
        h_1 = tf.nn.relu(tf.matmul(h_0, weights) + bias)

    with tf.name_scope(name='softmax_linear'):
        weights = tf.Variable(tf.truncated_normal(shape=[HIDDEN_UNITS[1], NUM_CLASSES],
                                                  stddev=0.1),
                              name='weights')
        bias = tf.Variable(tf.zeros(shape=[NUM_CLASSES]),
                           name='bias')
        logits = tf.matmul(h_1, weights) + bias
    return logits


def loss(logits, labels):
    labels = tf.to_int64(labels)
    cross_validation = tf.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                               logits=logits,
                                                               name='cross_entropy')

    return tf.reduce_mean(cross_validation, name='cross_entropy_reduced_mean')


def training(loss, learning_rate):
    # Add a scalar summary for the snapshot loss
    tf.summary.scalar('loss', loss)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

    global_step = tf.Variable(0,
                              trainable=False,
                              name='global_step')

    train_operation = optimizer.minimize(loss,
                                         global_step=global_step)

    return train_operation


def evaluation(logits, labels):
    return tf.reduce_sum(tf.cast(tf.nn.in_top_k(predictions=logits,
                                                targets=tf.argmax(tf.to_int64(labels), 1),
                                                k=1),
                                 tf.int32))


data_set = input_data.read_data_sets("MNIST_data/", one_hot=True)




# Tell TensorFlow that the model will be built into the default Graph.
with tf.Graph().as_default():
    image_placeholder = tf.placeholder(tf.float32, shape=[BATCH_SIZE, INPUT_SIZE])
    label_placeholder = tf.placeholder(tf.float32, shape=[BATCH_SIZE, NUM_CLASSES])

    logits = inference(image_placeholder)

    loss = loss(logits, label_placeholder)

    train_op = training(loss, LEARNING_RATE)

    eval_correct = evaluation(logits, label_placeholder)

    summary = tf.summary.merge_all()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(logdir='logs', graph=sess.graph)
        sess.run(tf.global_variables_initializer())

        for step in range(MAX_STEPS):
            start_time = time.time()
            images_feed, labels_feed = data_set.train.next_batch(BATCH_SIZE)
            _, loss_value = sess.run([train_op, loss],
                                     feed_dict={image_placeholder: images_feed,
                                                label_placeholder: labels_feed})

            duration = time.time() - start_time

            if step % 100 == 0:
                print('Step: %5d, Loss: %7.5f , (%5.3f)' % (step, loss_value, duration))
                summary_str = sess.run(summary,
                                       feed_dict={image_placeholder: images_feed,
                                                  label_placeholder: labels_feed})
                summary_writer.add_summary(summary_str, global_step=step)
                summary_writer.flush()

            if (step + 1) % 1000 == 0 or (step + 1) == MAX_STEPS:
                checkpoint_file = os.path.join('logs', 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=step)

                print('Training Data Evaluation: ')
                # And run one epoch of eval.
                true_count = 0  # Counts the number of correct predictions.
                steps_per_epoch = len(data_set.train.images) // BATCH_SIZE
                num_examples = steps_per_epoch * BATCH_SIZE
                for s in range(steps_per_epoch):
                    images_feed, labels_feed = data_set.train.next_batch(BATCH_SIZE)
                    true_count += sess.run(eval_correct,
                                           feed_dict={image_placeholder: images_feed,
                                                      label_placeholder: labels_feed})
                precision = float(true_count) / num_examples
                print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
                      (num_examples, true_count, precision))

                print('Validation Data Evaluation: ')
                true_count = 0  # Counts the number of correct predictions.
                steps_per_epoch = len(data_set.validation.images) // BATCH_SIZE
                num_examples = steps_per_epoch * BATCH_SIZE
                for s in range(steps_per_epoch):
                    images_feed, labels_feed = data_set.validation.next_batch(BATCH_SIZE)
                    true_count += sess.run(eval_correct,
                                           feed_dict={image_placeholder: images_feed,
                                                      label_placeholder: labels_feed})
                precision = float(true_count) / num_examples
                print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
                      (num_examples, true_count, precision))

        print('Test Data Evaluation: ')
        true_count = 0  # Counts the number of correct predictions.
        steps_per_epoch = len(data_set.test.images) // BATCH_SIZE
        num_examples = steps_per_epoch * BATCH_SIZE
        for s in range(steps_per_epoch):
            images_feed, labels_feed = data_set.test.next_batch(BATCH_SIZE)
            true_count += sess.run(eval_correct,
                                   feed_dict={image_placeholder: images_feed,
                                              label_placeholder: labels_feed})
        precision = float(true_count) / num_examples
        print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
              (num_examples, true_count, precision))