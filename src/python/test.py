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
from PlateRecognize import PlateTrain

if __name__ == '__main__':
   pt = PlateTrain.PlateTrain()
   pt.train_isPlateRegion(True)

