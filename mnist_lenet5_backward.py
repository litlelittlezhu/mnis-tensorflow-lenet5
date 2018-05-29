#coding=utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_lenet5_forward
import os
import numpy as np

# 定义训练过程中的超参数
BATCH_SIZE = 100 #一个batch的数量
LEARNING_RATE_BASE = 0.005 #定义初始学习速率
LEARNING_RATE_DECAY = 0.99 #定义学习速率衰减率
REGULARIZER = 0.0001 #定义正则化系数
STEPS = 50000 #最大迭代次数
MOVING_AVERAGE_DECAY = 0.99 #定义滑动平均系数
MODEL_SAVE_PATH = '/home/zbf/mnist/model1'#定义模型保存路径
MODEL_NAME = 'mnist_model'#定义模型文件名字

#训练过程
def backward(mnist):
	x = tf.placeholder(tf.float32,[BATCH_SIZE,
	mnist_lenet5_forward.IMAGE_SIZE,
	mnist_lenet5_forward.IMAGE_SIZE,
	mnist_lenet5_forward.NUM_CHANNELS])
	
	y_= tf.placeholder(tf.float32,[None,mnist_lenet5_forward.OUTPUT_NODE])
	y = mnist_lenet5_forward.forward(x,True,REGULARIZER)# 调用前向传播网络得到维度为10 的 tensor
	global_step = tf.Variable(0,trainable=False)# 声明一个全局计数器,并输出化为 0
	
	# 先是对网络最后一层的输出 y 做 softmax,通常是求取输出属于某一类的概率,其实就是一个num_classes 大小的向量,
	# 再将此向量和实际标签值做交叉熵,需要说明的是该函数返回的是一个向量
	ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
	cem = tf.reduce_mean(ce)# 再对得到的向量求均值就得到 loss
	loss = cem+tf.add_n(tf.get_collection('losses'))#损失函数
    # 实现指数级的减小学习率,可以让模型在训练的前期快速接近较优解,又可以保证模型在训练后期不会有太大波动
	# 计算公式:decayed_learning_rate=learining_rate*decay_rate^(global_step/decay_steps)
	learning_rate = tf.train.exponential_decay(
		LEARNING_RATE_BASE,
		global_step,
		mnist.train.num_examples / BATCH_SIZE,
		LEARNING_RATE_DECAY,
		staircase=True)## 当 staircase=True 时,(global_step/decay_steps)则被转化为整数,以此来选择不同的衰减方式

	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)#梯度下降法权重更新

	ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)#定义滑动平均
	#tf.train.exponential_decay(learning_rate, global_, decay_steps, decay_rate, staircase=True/False)
	ema_op = ema.apply(tf.trainable_variables())#对所有变量实行滑动平均

        #确保train_step,ema_op按顺序都执行
	with tf.control_dependencies([train_step,ema_op]):
		train_op = tf.no_op(name='train')
	
	saver = tf.train.Saver()## 实例化一个保存和恢复变量的 saver
	
	with tf.Session() as sess:# 创建一个会话,并通过 python 中的上下文管理器来管理这个会话
		init_op = tf.global_variables_initializer()# 初始化计算图中的变量
		sess.run(init_op)
		
		ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)# 通过 checkpoint 文件定位到最新保存的模型
		if ckpt and ckpt.model_checkpoint_path:#断点续训功能
			saver.restore(sess,ckpt.model_checkpoint_path)# 加载最新的模型
		
		for i in range(STEPS):
			xs,ys = mnist.train.next_batch(BATCH_SIZE)

			reshape_xs = np.reshape(xs,(
			BATCH_SIZE,
			mnist_lenet5_forward.IMAGE_SIZE,
			mnist_lenet5_forward.IMAGE_SIZE,
			mnist_lenet5_forward.NUM_CHANNELS))

			#喂入训练图像和标签,开始训练
			_,loss_value ,step= sess.run([train_op,loss,global_step],feed_dict={x:reshape_xs,y_:ys})
			if i%1000==0:# 每迭代 1000 次打印 loss 信息,并保存最新的模型
				print('After %d trining step(s),loss on trining batch is %g'%(step,loss_value))
				saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)#保存训练模型

def main():
	mnist = input_data.read_data_sets('/home/zbf/mnist',one_hot=True)
	backward(mnist)
if __name__ == '__main__':
	main()

