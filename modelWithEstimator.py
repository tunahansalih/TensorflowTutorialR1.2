import tensorflow as tf
import numpy as np

feature_columns = [tf.feature_column.numeric_column("x", shape=[1])]

estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)
x_train = np.linspace(0, 100, 101)
y_train = np.linspace(10, 60, 101)

x_valid = np.linspace(50, 150, 101)
y_valid = np.linspace(35, 85, 101)

input_fn = tf.estimator.inputs.numpy_input_fn({"x":x_train},
                                              y=y_train,
                                              batch_size=4,
                                              num_epochs=None,
                                              shuffle=True)

train_input_fn = tf.estimator.inputs.numpy_input_fn({"x":x_train},
                                                    y=y_train,
                                                    batch_size=4,
                                                    num_epochs=1000,
                                                    shuffle=False)

valid_input_fn = tf.estimator.inputs.numpy_input_fn({"x":x_valid},
                                                    y=y_valid,
                                                    batch_size=4,
                                                    num_epochs=1000,
                                                    shuffle=False)

estimator.train(input_fn=input_fn, steps=1000)

train_metrics = estimator.evaluate(train_input_fn)
validation_metrics = estimator.evaluate(valid_input_fn)
print("Train metrics: %r" % train_metrics)
print("Validation metrics: %r" %validation_metrics)