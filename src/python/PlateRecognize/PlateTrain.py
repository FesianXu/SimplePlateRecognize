# !/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'FesianXu'
__date__ = '2017/5/21'
__version__ = 'version 0.1'

'''
负责训练模型，用于判断是否是车牌，是否是字符
'''

from svmutil import *
import numpy as np
import PlateRecognize.FeatureExtraction as FeatureExtraction
import os
import cv2
import random
import tensorflow as tf
import time


class PlateTrain(object):
    '''
    训练相关的分类器模型， 用于判断是否是车牌，是否是字符， 字符识别等
    version 0.1 使用SVM判断车牌字符，使用卷积神经网络CNN识别字符
    '''
    __is_plate_model_save_path = '/src/python/train_data/is_plate/is_plate_svm_model.model'
    __predict_char_model_save_path = '/src/python/train_data/predict_chars/predict_chars_cnn_model.ckpt'
    __number_samples_mat_path = '/res/trainning_set/normalized/numbers/'
    __alphabet_samples_mat_path = '/res/trainning_set/normalized/alphabet/'
    __number_each_path = []
    __alphabet_each_path = []
    __number_list_mat = []
    __alphabet_list_mat = []
    __project_root_path = u''  # 绝对项目路径
    __isSet = False  # 是否已经设置了绝对项目路径
    __n_classes = 34  # 包括字母和数字
    __n_input = 32*64
    __out_loop_limits = 1
    __train_inner_times = 1
    __cnn_dropout = 0.75
    __learning_rate = 0.001
    isplate_svm = FeatureExtraction.FeatureExtraction()

    def __init__(self):
        if self.__isSet is not True:
            BASE_DIR = self.__project_root_path = os.path.dirname(__file__)
            self.__project_root_path = os.path.dirname(os.path.dirname(os.path.dirname(BASE_DIR)))
            self.__is_plate_model_save_path = self.__project_root_path+self.__is_plate_model_save_path
            self.__predict_char_model_save_path = self.__project_root_path+self.__predict_char_model_save_path
            self.__number_samples_mat_path = self.__project_root_path+self.__number_samples_mat_path
            self.__alphabet_samples_mat_path = self.__project_root_path+self.__alphabet_samples_mat_path
            self.__number_each_path = [self.__number_samples_mat_path+x+'/' for x in os.listdir(self.__number_samples_mat_path)]
            self.__alphabet_each_path = [self.__alphabet_samples_mat_path+x+'/' for x in os.listdir(self.__alphabet_samples_mat_path)]
            self.__number_list_mat = self.__divideTestAndTrain(self.__number_each_path)
            self.__alphabet_list_mat = self.__divideTestAndTrain(self.__alphabet_each_path)
        self.__isSet = True


    # TODO(FesianXu) 改进其效果，目前交叉检验只有62%的精确度
    def train_isPlateRegion(self, isSaveRaw=False, isSaveModel=True):
        '''
        :: 判断是否是车牌区域的SVM模型
        :: 目前效果不好
        :param isSaveRaw: 是否要保存原始的特征数据
        :param isSaveModel: 是否要保存训练好的svm model
        :return: svm model
        '''
        pos = self.isplate_svm.getPlateFeature(self.isplate_svm.getIsPlate_PosPath())
        neg = self.isplate_svm.getPlateFeature(self.isplate_svm.getIsPlate_NegPath())
        feature = pos+neg
        label = [1.0]*len(pos)+[-1.0]*len(neg)
        model = svm_train(label, feature)
        if isSaveModel:
            # 需要更改libsvm中的svmutil中的libsvm.svm_save_model(model_file_name.encode(), model)的encode()为'gbk'模式
            svm_save_model(self.__is_plate_model_save_path, model)
        if isSaveRaw:
            np.array(feature).tofile(self.isplate_svm.getIsPlate_FeatureMatPath())
            np.array(label).tofile(self.isplate_svm.getIsPlate_LabelMatPath())
        else:
            pass
        return model


    def load_isPlate_svm_model(self):
        '''
        :: 加载是否是车牌的svm模型
        :return: svm模型
        '''
        model = svm_load_model(self.__is_plate_model_save_path)
        return model


    def __divideTestAndTrain(self, each_path, test_batch=50):
        '''
        :: 随机在一个样本集中划分测试集和训练集
        :param each_path: 需要划分的文件夹list
        :param test_batch: 测试样本的尺寸
        :return: list_mat test list和train list， 只是名字而已，而且没有根目录
        '''
        list_mat = []
        for each in each_path:
            dirlist = os.listdir(each)
            test_list = random.sample(dirlist, test_batch)
            train_list = [x for x in dirlist if x not in test_list]
            list_mat.append([test_list, train_list])
        return list_mat

    def __getFoldersImages(self, root_path, path_list, batch_size):
        if batch_size < 0:
            dirlist = path_list
            batch_size = len(path_list)
        else:
            dirlist = random.sample(path_list, batch_size)
        sample_list = []
        for each in dirlist:
            file_name = root_path+each
            img = cv2.imdecode(np.fromfile(file_name, dtype=np.uint8), -1)
            img = img.reshape(1, img.size)
            sample_list.append(img)
        return sample_list, batch_size


    def __getSamplesMat(self, type, batch_size):
        cx_img, cx_len = [], []
        for i in range(self.__n_classes):
            if i < 10:
                ci_img, ci_len = self.__getFoldersImages(self.__number_each_path[i], self.__number_list_mat[i][type], batch_size)
            else:
                ci_img, ci_len = self.__getFoldersImages(self.__alphabet_each_path[i-10], self.__alphabet_list_mat[i-10][type], batch_size)
            cx_img.append(ci_img)
            cx_len.append(ci_len)
        labels_mat = np.zeros([sum(cx_len), self.__n_classes], dtype=np.float32)
        samples_mat = np.zeros([sum(cx_len), self.__n_input], dtype=np.float32)
        for i in range(self.__n_classes):
            labels_mat[i*batch_size:(i+1)*batch_size, i] = 1
            samples_mat[i*batch_size:(i+1)*batch_size, :] = cx_img[i]
        return samples_mat, labels_mat


    def train_predict_chars(self):
        __fc_inner_cell = 1024
        # Store layers weight & bias
        weights = {
            # 5x5 conv, 1 input, 32 outputs
            'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
            # 5x5 conv, 32 inputs, 64 outputs
            'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
            # fully connected, 7*7*64 inputs, 1024 outputs
            'wd1': tf.Variable(tf.random_normal([16*8*64, __fc_inner_cell])),
            # 1024 inputs, 10 outputs (class prediction)
            'out': tf.Variable(tf.random_normal([__fc_inner_cell, self.__n_classes]))
        }

        biases = {
            'bc1': tf.Variable(tf.random_normal([32])),
            'bc2': tf.Variable(tf.random_normal([64])),
            'bd1': tf.Variable(tf.random_normal([__fc_inner_cell])),
            'out': tf.Variable(tf.random_normal([self.__n_classes]))
        }
        # tf Graph input
        x = tf.placeholder(tf.float32, [None, self.__n_input])
        y = tf.placeholder(tf.float32, [None, self.__n_classes])
        keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

        # Create some wrappers for simplicity
        def conv2d(x, W, b, strides=1):
            # Conv2D wrapper, with bias and relu activation
            x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
            x = tf.nn.bias_add(x, b)
            return tf.nn.relu(x)

        def maxpool2d(x, k=2):
            # MaxPool2D wrapper
            return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                                  padding='SAME')

        # Create model
        def conv_net(x, weights, biases, dropout):
            x = tf.reshape(x, shape=[-1, 64, 32, 1])
            # Convolution Layer
            conv1 = conv2d(x, weights['wc1'], biases['bc1'])
            # Max Pooling (down-sampling)
            conv1 = maxpool2d(conv1, k=2)
            # Convolution Layer
            conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
            # Max Pooling (down-sampling)
            conv2 = maxpool2d(conv2, k=2)
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

        # Construct model
        pred = conv_net(x, weights, biases, keep_prob)
        # Define loss and optimizer
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.__learning_rate).minimize(cost)
        # Evaluate model
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        # Initializing the variables
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        print('begin to train CNN')
        time_train_begin = time.clock()
        with tf.Session() as sess:
            sess.run(init)

            for i in range(self.__out_loop_limits):
                avg_cost = 0
                for j in range(self.__train_inner_times):
                    train_x, train_y = self.__getSamplesMat(1, 100)
                    sess.run(optimizer, feed_dict={x: train_x, y: train_y,
                                                   keep_prob: self.__cnn_dropout})
                    if True:
                        loss, acc = sess.run([cost, accuracy], feed_dict={x: train_x,
                                                                          y: train_y,
                                                                          keep_prob: 1.})
                        print("i = " + str(i)+ '  j = '+str(j)+ ", Minibatch Loss= " + \
                              "{:.6f}".format(loss) + ", Training Accuracy= " + \
                              "{:.5f}".format(acc))
            time_train_end = time.clock()
            saver_path = saver.save(sess, 'G:\cnn_model\predict_chars_cnn_model.ckpt')
            print("Optimization Finished!")
            test_x, test_y = self.__getSamplesMat(0, 50)
            time_test_begin = time.clock()
            print("Testing Accuracy:", \
                        sess.run(accuracy, feed_dict={x: test_x,
                                                      y: test_y,
                                                      keep_prob: 1.}))
            time_test_end = time.clock()
            print('train time cost = ', time_train_end-time_train_begin)
            print('test time cost = ', time_test_end-time_test_begin)


########################################################################################################################


if __name__ == '__main__':
    tr = PlateTrain()
    tr.train_predict_chars()
    pass