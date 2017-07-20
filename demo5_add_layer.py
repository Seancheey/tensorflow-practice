import tensorflow as tf
import numpy as np


### 训练多层神经网络的例子

# function that adds one layer of neuron
def add_layer(inputs, in_size, out_size, activation_function=lambda _: _):
	Weights = tf.Variable(tf.random_normal([in_size, out_size]))
	Biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
	Wx_plus_b = tf.matmul(inputs, Weights) + Biases
	return activation_function(Wx_plus_b)


# create a x-y relationship of y = x**2 - 0.5 with some noise
x_data = np.linspace(-1, 1, 300, dtype=np.float32)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise

# add the first layer
layer1 = add_layer(x_data, 1, 10, activation_function=tf.nn.relu)
# add the second layer, with layer1 as input
prediction = add_layer(layer1, 10, 1)

# estimate loss
loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_data - prediction), reduction_indices=[1]))

# train to minimize loss
train_step = tf.train.GradientDescentOptimizer(0.03).minimize(loss)

# start session
with tf.Session() as session:
	session.run(tf.global_variables_initializer())
	for i in range(1000):
		session.run(train_step)
		if i % 10 == 0:
			print(session.run(loss))
