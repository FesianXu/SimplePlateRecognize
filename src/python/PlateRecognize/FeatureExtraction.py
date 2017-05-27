# !/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'FesianXu'
__date__ = '2017/5/23'
__version__ = 'version 0.1'

'''
用于车牌和车牌字符的特征提取
'''

import cv2
import os
import numpy as np
from svmutil import *
import test

class FeatureExtraction(object):
    '''
    用于提取车牌和车牌中字符的特征，以供交于分类器训练学习
    '''

    __is_plate_pos_path = '/res/trainning_set/is_plate/pos_samples/'
    __is_plate_neg_path = '/res/trainning_set/is_plate/neg_samples/'
    __is_plate_feature_mat = '\src\python\train_data\is_plate\is_plate_feature_mat.raw'
    __is_plate_label_mat = '/src/python/train_data/is_plate/is_plate_label_mat.raw'
    __plate_norm_width = 180  # 车牌标准化长度
    __plate_norm_height = 50  # 车牌标准宽度
    __max_BinImage = 1  # 二值图中的最大值
    __root_path = ''

    def __init__(self, plate_width=180, plate_height=50):
        self.__plate_norm_width = plate_width
        self.__plate_norm_height = plate_height
        self.__root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath('FeatureExtraction.py'))))
        self.__is_plate_pos_path = os.listdir(self.__root_path+self.__is_plate_pos_path)
        self.__is_plate_neg_path = os.listdir(self.__root_path+self.__is_plate_neg_path)
        self.__is_plate_feature_mat = os.listdir(self.__root_path+self.__is_plate_feature_mat)
        self.__is_plate_label_mat = os.listdir(self.__root_path+self.__is_plate_label_mat)

        pass


    def getPlateFeature(self, path):
        '''
        :: 获得车牌区域的特征矩阵
        :param path: 车牌或者非车牌区域的根目录
        :return: 特征向量list
        '''
        dir_list = os.listdir(path)
        feature_mat = []
        for ind, each_img in enumerate(dir_list):
            file_name = path+each_img
            img = cv2.imread(file_name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (self.__plate_norm_width, self.__plate_norm_height))
            _, img = cv2.threshold(img, 120, 1, cv2.THRESH_BINARY)
            feature = img.reshape(img.size, self.__max_BinImage).transpose().astype(float).tolist()[0]
            feature_mat.append(feature)
        return feature_mat


    def getIsPlate_PosPath(self):
        return self.__is_plate_pos_path

    def getIsPlate_NegPath(self):
        return self.__is_plate_neg_path

    def getIsPlate_FeatureMatPath(self):
        return self.__is_plate_feature_mat

    def getIsPlate_LabelMatPath(self):
        return self.__is_plate_label_mat


########################################################################################################################

if __name__ == '__main__':
    # fe = FeatureExtraction()
    # fe.isPlate_FeatureExtract()
    print(os.path.abspath('PlateTrain.py'))


    print()