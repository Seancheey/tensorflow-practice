import tensorflow as tf
import subprocess
import numpy as np


### define a neuron network that can be displayed in tensorboard --- a visualization tool supported by tensorflow


# define a add layer function
def add_layer(inputs, in_size, out_size, activation_function=tf.nn.relu):
	# define name scope to help visualizing neuron layers
	with tf.name_scope('layer'):
		with tf.name_scope('weights'):
			Weights = tf.Variable(tf.random_normal([in_size, out_size]))
		with tf.name_scope('biases'):
			Biases = tf.Variable(tf.random_normal([1, out_size]))
		with tf.name_scope('Wx_plus_b'):
			Wx_plus_b = tf.add(tf.matmul(inputs, Weights), Biases)
		return activation_function(Wx_plus_b)


# define name scope to help visualizing input block
with tf.name_scope('inputs'):
	# define inputs with their names
	xs = tf.placeholder(tf.float32, [None, 1], 'x-input')
	ys = tf.placeholder(tf.float32, [None, 1], 'y-input')

layer1 = add_layer(xs, 1, 10)
layer2 = add_layer(layer1, 10, 1)

with tf.name_scope('loss'):
	loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - layer2), reduction_indices=[1]))

with tf.name_scope('train'):
	train_step = tf.train.GradientDescentOptimizer(0.03).minimize(loss)

# create a x-y relationship of y = x**2 - 0.5 with some noise
x_data = np.linspace(-1, 1, 300, dtype=np.float32)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise


with tf.Session() as session:
	session.run(tf.global_variables_initializer())
	session.run(train_step, feed_dict={xs: x_data, ys: y_data})
	# write the graph summary into logs/
	writer = tf.summary.FileWriter('logs/', session.graph)
	# execute command to open a 6006 port in localhost that visualize data
	subprocess.getoutput('tensorboard --logdir logs')
	# after that, type 0.0.0.0:6006/ or localhost:6006/ in any browser to check tensorboard