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

# import tensorflow as tf




class PlateTrain(object):
    '''
    训练相关的分类器模型， 用于判断是否是车牌，是否是字符， 字符识别等
    '''
    __is_plate_model_save_path = '../train_data/is_plate/is_plate_svm_model.model'
    isplate_svm = FeatureExtraction.FeatureExtraction()

    def __init__(self):
        pass

    @test.timeit
    def train_isPlateRegion(self, isSaveRaw=False, isSaveModel=True):
        '''
        :: 判断是否是车牌区域的SVM模型
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
            svm_save_model(self.__is_plate_model_save_path, model)
        if isSaveRaw:
            np.array(feature).tofile(self.isplate_svm.getIsPlate_FeatureMatPath())
            np.array(label).tofile(self.isplate_svm.getIsPlate_LabelMatPath())
        else:
            pass
        return model






########################################################################################################################


if __name__ == '__main__':
    tr = PlateTrain()
    tr.train_isPlateRegion(True)
    pass