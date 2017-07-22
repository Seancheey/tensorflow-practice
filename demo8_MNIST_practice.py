import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('logs/', one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y_predict = tf.matmul(x, W) + b


cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_predict), reduction_indices=[1]))  # loss

train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

with tf.Session() as session:
	session.run(tf.global_variables_initializer())
	for _ in range(1000):
		batch_xs, batch_ys = mnist.train.next_batch(100)
		session.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
	# evaluate model
	correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_predict, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	print(session.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
