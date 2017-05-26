# !/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'FesianXu'
__date__ = '2017/5/23'
__version__ = ''

'''
用于实验纪录文档生成和实验图像保存等
'''

__saved_chars_path = ''


def saveChars(func):
    def wrapper(*args, **kwargs):
        roi_set = func(*args, **kwargs)

        return roi_set
    return wrapper

class PlateLogger(object):

    __test_record_path = r'../../../test_record/txt_files'  # 实验记录文件夹


    def __init__(self):
        pass