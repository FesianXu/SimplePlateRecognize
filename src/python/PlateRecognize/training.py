# !/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'FesianXu'
__date__ = '2017/5/21'
__version__ = ''

'''
负责训练模型，用于判断是否是车牌，是否是字符
'''

from svmutil import *
# import tensorflow as tf

def train_isPlateRegion():
    pass


def train_isChar():
    pass



########################################################################################################################

def test_svm():
    data_path = 'G:/libsvm-3.22/heart_scale'
    y, x = svm_read_problem(data_path)
    m = svm_train(y[:200], x[:200], '-c 4')
    p_label, p_acc, p_val = svm_predict(y[200:], x[200:], m)

if __name__ == '__main__':
    test_svm()
    pass