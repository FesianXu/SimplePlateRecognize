# !/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'FesianXu'
__date__ = '2017/5/20'
__version__ = ''


import time
import functools
from svmutil import *

def timeit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        begin = time.clock()
        res = func(*args, **kwargs)
        end = time.clock()
        print('time cost = ', end-begin, 'in function = ', func.__name__)
        return res
    return wrapper



########################################################################################################################
'''
optimization finished, #iter = 480 迭代次数
nu = 0.909091 SVC,one-class-SVM,SVR参数
obj = -108.333321 二次规划的最小值
rho = -0.166667 决策函数常数项
nSV = 220 支持向量数
nBSV = 100 边界上支持向量数
Total nSV = 220 支持向量总数
Accuracy = 100% (220/220) (classification) 分类精度
'''

from PlateRecognize import PlateTrain
import cv2
import numpy as np

if __name__ == '__main__':
   pt = PlateTrain.PlateTrain()
   # pt.train_isPlateRegion(False, True)
   m = pt.load_isPlate_svm_model()
   path = 'C:/2.jpg'
   img = cv2.imread(path, 0)
   _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
   f = img.reshape(1, 9000).astype(float).tolist()
   y = [1]
   l, a, v = svm_predict(y, f, m)
   print(l[0])
   print(a)
   print(v)

