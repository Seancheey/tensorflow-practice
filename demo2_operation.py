import tensorflow as tf

### 使用tensorflow进行运算的方法
### 主要注意tensorflow的运算是先定义运算方法再运算
### 有些类似lazy evaluation

# 创建两个常量矩阵
matrix1 = tf.constant([[3, 3]])
matrix2 = tf.constant([[2], [2]])

# 进行叉乘运算操作，但是运算并没有实际被进行
product = tf.matmul(matrix1, matrix2)
print(product)

# 打开会话，真正运行操作
with tf.Session() as session:
	result = session.run(product)
	print(result)
