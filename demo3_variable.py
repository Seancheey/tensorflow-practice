import tensorflow as tf

### 创建变量时需要注意的地方，先要初始化变量操作


# 创建一个叫做my_var的变量
state = tf.Variable(371, name="my_var")

# 创建一个常量
one = tf.constant(1)

# 常量和变量相加
new_value = tf.add(state, one)

# 将变量的值赋予两者相加值
update = tf.assign(state, new_value)

# 开始会话
with tf.Session() as session:
	# 使用变量时必须要先进行初始化变量的操作
	session.run(tf.global_variables_initializer())
	# 进行5次增1操作并输出
	for _ in range(5):
		session.run(update)
		print(session.run(state))
