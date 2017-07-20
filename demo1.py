import tensorflow as tf
import numpy as np

# 使用 NumPy 生成100个数据
# 使得 x-y 的对应关系为 y = 0.1 * x + 0.5
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.5

# 创建仅一层神经网络的权重和偏值
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
Biases = tf.Variable(tf.zeros([1]))
Y = Weights * x_data + Biases

# 计算误差
loss = tf.reduce_mean(tf.square(Y - y_data))

# 修正误差，给定学习速度
optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

# 创建会话
session = tf.Session()
session.run(init)

# 运行，每20次运行输出权重和偏值
# 运行越多次权重/偏值应越接近真实结果 (y = 0.1 * x + 0.5)
for step in range(3000):
	session.run(train)
	if step % 20 == 0:
		print(step, session.run(Weights), session.run(Biases))