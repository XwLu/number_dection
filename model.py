# -*- coding: utf-8 -*-
#coding=utf-8

import input_data
import tensorflow as tf
import numpy as np
import cv2


"""权重初始化"""
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


"""卷积和池化"""
"""strides代表1步长，padding代表添加0边距"""
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

class NumberDetection():
    def __init__(self):

        self.sess = tf.InteractiveSession()

        """placeholder占位符"""
        self.x = tf.placeholder("float", shape=[None, 784])
        self.y_ = tf.placeholder("float", shape=[None, 10])

        """第一层卷积"""
        self.W_conv1 = weight_variable([5, 5, 1, 32])
        self.b_conv1 = bias_variable([32])
        # 图片像素28 × 28，最后一个“1”代表单通道颜色
        self.x_image = tf.reshape(self.x, [-1, 28, 28, 1])
        self.h_conv1 = tf.nn.relu(conv2d(self.x_image, self.W_conv1) + self.b_conv1)
        self.h_pool1 = max_pool_2x2(self.h_conv1)

        """第二层卷积"""
        self.W_conv2 = weight_variable([5, 5, 32, 64])
        self.b_conv2 = bias_variable([64])
        self.h_conv2 = tf.nn.relu(conv2d(self.h_pool1, self.W_conv2) + self.b_conv2)
        self.h_pool2 = max_pool_2x2(self.h_conv2)

        """密集连接层"""
        self.W_fc1 = weight_variable([7 * 7 * 64, 1024])
        self.b_fc1 = bias_variable([1024])
        self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, 7 * 7 * 64])
        self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, self.W_fc1) + self.b_fc1)

        """Dropout层"""
        self.keep_prob = tf.placeholder("float")
        self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)

        """输出层"""
        self.W_fc2 = weight_variable([1024, 10])
        self.b_fc2 = bias_variable([10])

        self.y_conv = tf.nn.softmax(tf.matmul(self.h_fc1_drop, self.W_fc2) + self.b_fc2)

        self.saver = tf.train.Saver() # default to save all variables -- in this case, all weights and bias

    def train(self, learning_rate, step, dropout):
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        """训练"""
        cross_entropy = -tf.reduce_sum(self.y_ * tf.log(self.y_conv))
        tf.summary.scalar("loss", cross_entropy)
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
        # train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        self.sess.run(tf.global_variables_initializer())

        for i in range(step):
            batch = mnist.train.next_batch(50)
            """keep_prob是Dropout比例"""
            train_step.run(feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: dropout})
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    self.x: batch[0], self.y_: batch[1], self.keep_prob: 1.0})
                print "step %d, training accuracy %g" % (i, train_accuracy)

        """评估"""
        print "test accuracy %g" % accuracy.eval(feed_dict={
            self.x: mnist.test.images, self.y_: mnist.test.labels, self.keep_prob: 1.0})

    def saveWeights(self, checkpoint_dir):
        self.saver.save(self.sess, checkpoint_dir+'/model.ckpt')
        print 'model saved at '+checkpoint_dir+'/model.ckpt'

    def loadWeights(self, checkpoint_dir):
        self.saver.restore(self.sess, checkpoint_dir+'/model.ckpt')
        print 'model reload successfully!'

    def picturePredict(self, image_path):
        img = cv2.imread(image_path, 0)

        image = np.ndarray(shape=(784))

        for i in range(len(img)):
            for j in range(len(img[0])):
                image[i * 28 + j] = img[i][j] / 255.0

        images = []
        images.append(image)

        images = np.asarray(images)

        """1张图片包括图片的标签"""
        #image_set = self.mnist.train.next_batch(1)
        #image = image_set[0]
        #print(image)
        #label = image_set[1]
        #label_num = self.sess.run(tf.argmax(label[0]))
        #print(label_num)
        result = self.sess.run(self.y_conv, feed_dict={self.x: images, self.keep_prob: 1.0})
        #print(result[0])
        number = self.sess.run(tf.argmax(result[0]))
        print(number)

    def cameraPredict(self):
        cap = cv2.VideoCapture(0)
        image = np.ndarray(shape=(784))

        while (1):
            ret, frame = cap.read()

            gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)

            """先膨胀图像"""
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            dilated = cv2.dilate(thresh, kernel)
            """腐蚀图像"""
            NpKernel = np.uint8(np.ones((3, 3)))
            Nperoded = cv2.erode(dilated, NpKernel)

            for i in range(len(Nperoded)):
                for j in range(len(Nperoded[0])):
                    Nperoded[i][j] = 255 - Nperoded[i][j]

            img = cv2.resize(Nperoded, dsize=(28, 28))

            for i in range(28):
                for j in range(28):
                    image[i * 28 + j] = img[i][j] / 255.0

            images = []
            images.append(image)

            images = np.asarray(images)

            result = self.sess.run(self.y_conv, feed_dict={self.x: images, self.keep_prob: 1.0})
            number = self.sess.run(tf.argmax(result[0]))
            print(number)

            cv2.imshow("capture", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break