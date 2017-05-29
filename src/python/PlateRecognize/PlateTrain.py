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
import test
import os


class PlateTrain(object):
    '''
    训练相关的分类器模型， 用于判断是否是车牌，是否是字符， 字符识别等
    '''
    __is_plate_model_save_path = '/src/python/train_data/is_plate/is_plate_svm_model.model'
    __project_root_path = u''  # 绝对项目路径
    __isSet = False  # 是否已经设置了绝对项目路径
    isplate_svm = FeatureExtraction.FeatureExtraction()

    def __init__(self):
        if self.__isSet is not True:
            BASE_DIR = self.__project_root_path = os.path.dirname(__file__)
            self.__project_root_path = os.path.dirname(os.path.dirname(os.path.dirname(BASE_DIR)))
            self.__is_plate_model_save_path = self.__project_root_path+self.__is_plate_model_save_path
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





########################################################################################################################


if __name__ == '__main__':
    tr = PlateTrain()
    tr.train_isPlateRegion(False, True)
    pass