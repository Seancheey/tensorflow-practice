import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# practice using neuron network to recognize numbers in MNIST dataset. This is a classification problem
# if you don't know what MNIST data is, check this out: http://yann.lecun.com/exdb/mnist/

# read MNIST data in to logs/ directory.
# A one-hot vector is a vector which is 0 in most dimensions, and 1 in a single dimension.
# In this case, the nth digit will be represented as a vector which is 0 in most dimensions, and 1 in a single dimension
# explanation retrieved from https://www.tensorflow.org/get_started/mnist/beginners
mnist = input_data.read_data_sets('logs/', one_hot=True)

# create two placeholers for x as input data and y as its number ranging from 0-9
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# create weights and biases
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# softmax is a type of activation function that makes all elements in a list lie in range [0,1] and add up to 1.
y_predict = tf.nn.softmax(tf.matmul(x, W) + b)

# calculate cross entropy to ease minimize the loss of this classification problem
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_predict), reduction_indices=[1]))  # loss

# train
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

with tf.Session() as session:
	session.run(tf.global_variables_initializer())
	for _ in range(1000):
		batch_xs, batch_ys = mnist.train.next_batch(100)
		session.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
	# evaluate model
	correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_predict, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	print("accuracy:", session.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
