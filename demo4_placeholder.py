import tensorflow as tf

### python中占位符的作用主要是用来引入外部数据

# 定义两个占位符
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

# 定义相乘操作
multiply = tf.multiply(input1, input2)

with tf.Session() as session:
	# 注意使用占位符的时候必须要把具体值传入，否则会报错
	result = session.run(multiply, feed_dict={input1: [2.], input2: [5.]})
	print(result)
