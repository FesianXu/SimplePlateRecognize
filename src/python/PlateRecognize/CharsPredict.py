# !/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'FesianXu'
__date__ = '2017/6/1'
__version__ = ''

import cv2
import tensorflow as tf
import time


fc_inner_cell = 1024  # 全连接层的隐藏层神经元大小
n_classes = 34
n_input = 32*64
# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([16*8*64, fc_inner_cell])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([fc_inner_cell, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([fc_inner_cell])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


# Create some wrappers for simplicity
def __conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def __maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

# Create model
def __conv_net(x, weights, biases, dropout):
    x = tf.reshape(x, shape=[-1, 64, 32, 1])
    # Convolution Layer
    conv1 = __conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = __maxpool2d(conv1, k=2)
    # Convolution Layer
    conv2 = __conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = __maxpool2d(conv2, k=2)
    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)
    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

__is_load_cnn_model = False
sess_predict = tf.Session()

pred = __conv_net(x, weights, biases, keep_prob)
op = tf.argmax(pred, 1)

if __is_load_cnn_model is False:
    saver = tf.train.Saver()
    saver.restore(sess_predict, r'G:\cnn_model\predict_chars_cnn_model.ckpt')
    __is_load_cnn_model = True


def __Id2Chars(res_list):
    res = []
    for each in res_list:
        res_tmp = None
        if 0 <= each <= 9:
            begin = ord('0')
            res_tmp = chr(begin+each)
        elif 10 <= each <= 17:
            begin = ord('A')
            res_tmp = chr(begin+each-10)
        elif 18 <= each <= 22:
            begin = ord('J')
            res_tmp = chr(begin+each-18)
        elif 23 <= each <= 33:
            begin = ord('P')
            res_tmp = chr(begin+each-23)
        res.append(res_tmp)
    return res


def predict_chars(img_list):
    '''
    :: 识别字符
    :param img_list: 字符列表
    :return: 识别结果
    '''
    res = []
    for each in img_list:
        feature = each.reshape(1, each.size)
        res_tmp = sess_predict.run(op, feed_dict={x: feature, keep_prob: 1.0})
        res.append(res_tmp[0])
    res = __Id2Chars(res)
    return res

def close_sess():
    sess_predict.close()


########################################################################################################################

if __name__ == '__main__':
    img1 = cv2.imread('C:/1.jpg', -1)


    img_list = [img1, img1, img1, img1, img1, img1, img1]

    c = 10
    timebegin = time.clock()
    for i in range(c):
        print(predict_chars(img_list))
    timeend = time.clock()

    print('cost = ', (timeend-timebegin)/c)
    print('total = ', timeend-timebegin)