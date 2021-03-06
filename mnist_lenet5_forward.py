#coding=utf-8
import tensorflow as tf

#设定神经网络参数
#定义神经网络可以接受的图片的尺寸和通道数

IMAGE_SIZE=28
NUM_CHANNELS=1

#定义第一层卷积核的大小和个数
CONV1_SIZE=5
CONV1_KERNEL_NUM=32

#定义第二层卷积核的大小和个数
CONV2_SIZE=5
CONV2_KERNEL_NUM=64

#定义第三层全连接层的神经元个数
FC_SIZE=512
#定义第四层全连接层的神经元个数
OUTPUT_NODE=10

#定义初始化网络权重
def get_weight(shape,regularizer):
	w = tf.Variable(tf.truncated_normal(shape,stddev=0.1))
	if regularizer!=None:
		tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(regularizer)(w))
	return w
#定义初始化偏置项函数
def get_bias(shape):
	b=tf.Variable(tf.zeros(shape))
	return b

#定义卷积计算函数
def conv2d(x,w):
	return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')
#定义池化计算函数
def max_pool_2x2(x):
	# ksize 表示池化过滤器的边长为 2,strides 表示过滤器移动步长是 2,'SAME'提供使用全 0 填充
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
#定义前向传播过程
def forward(x,train,regularizer):
	#实现第一层卷积层的前向传播
	conv1_w = get_weight([CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,CONV1_KERNEL_NUM],regularizer)
	conv1_b = get_bias([CONV1_KERNEL_NUM])#初始化偏置项
	conv1 = conv2d(x,conv1_w)#实现卷积运算
	relu1 = tf.nn.relu(tf.nn.bias_add(conv1,conv1_b))#对卷积后的输出添加偏置,并过 relu 非线性激活函数
	pool1 = max_pool_2x2(relu1) # 将激活后的输出进行最大池化
	
	#实现第一层卷积层的前向传播
	conv2_w = get_weight([CONV2_SIZE,CONV2_SIZE,CONV1_KERNEL_NUM,CONV2_KERNEL_NUM],regularizer)
	conv2_b = get_bias([CONV2_KERNEL_NUM])#初始化偏置项
	conv2 = conv2d(pool1,conv2_w)#实现卷积运算
	relu2 = tf.nn.relu(tf.nn.bias_add(conv2,conv2_b))#对卷积后的输出添加偏置,并过 relu 非线性激活函数
	pool2 = max_pool_2x2(relu2) # 将激活后的输出进行最大池化
	
	# 将上一池化层的输出 pool2(矩阵)转化为下一层全连接层的输入格式(向量)
	pool_shape=pool2.get_shape().as_list()#得到 pool2 输出矩阵的维度,并存入 list 中,注意 pool_shape[0]是一个 batch 的值
	nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]# 从 list 中依次取出矩阵的长宽及深度,并求三者的乘积就得到矩阵被拉长后的长度
	reshaped = tf.reshape(pool2,[pool_shape[0],nodes])#将 pool2 转换为一个 batch 的向量再传入后续的全连接
	# 实现第三层全连接层的前向传播过程
	fc1_w = get_weight([nodes,FC_SIZE],regularizer)#初始化全连接层的权重，并加入正则化
	fc1_b = get_bias([FC_SIZE])#初始化去全连接层偏置项
	# 将转换后的 reshaped 向量与权重 fc1_w 做矩阵乘法运算,然后再加上偏置,最后再使用 relu 进行激活
	fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_w) + fc1_b)
	# 如果是训练阶段,则对该层输出使用 dropout,也就是随机的将该层输出中的一半神经元置为无效,是为了避免过拟合而设置的,一般只在全连接层中使用
	if train:
		fc1 = tf.nn.dropout(fc1,0.5)
	# 实现第四层全连接层的前向传播过程,并初始化全连接层对应的变量
	fc2_w = get_weight([FC_SIZE,OUTPUT_NODE],regularizer)
	fc2_b = get_bias([OUTPUT_NODE])
	y = tf.matmul(fc1,fc2_w)+fc2_b
	return y









