import TensorFlow.demo5_add_layer as demo5
import matplotlib.pyplot as plt
import tensorflow as tf

### 使用demo5的神经网络做的简单图形化操作


with tf.Session() as session:
	session.run(tf.global_variables_initializer())

	# create a graph to visualize the graph
	fig = plt.figure()
	axis = fig.add_subplot(1, 1, 1)
	# add the scatter data into graph
	axis.scatter(demo5.x_data, demo5.y_data)
	# make the plot change with respect to time
	plt.ion()
	# show the plot GUI
	plt.show()

	# train for 5000 times
	for i in range(5000):
		session.run(demo5.train_step)

		if i % 10 == 0:
			try:
				# remove previously drawn line
				axis.lines.remove(lines[0])
			except Exception:
				pass
			# draw the prediction line
			prediction = session.run(demo5.prediction)
			lines = axis.plot(demo5.x_data, prediction, 'r-', lw=5)
			# pause to slow down the process
			plt.pause(0.02)
