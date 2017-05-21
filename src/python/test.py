# !/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'FesianXu'
__date__ = '2017/5/20'
__version__ = ''


import cv2
import numpy as np
import time
import functools

def timeit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        begin = time.clock()
        res = func(*args, **kwargs)
        end = time.clock()
        print('time cost = ', end-begin, 'in function = ', func.__name__)
        return res
    return wrapper



if __name__ == '__main__':
    pass