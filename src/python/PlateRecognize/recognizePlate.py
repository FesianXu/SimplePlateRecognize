# !/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'FesianXu'
__date__ = '2017/6/1'
__version__ = 'version 0.1'


import PlateRecognize.PlateSegment as PlateSegment
import PlateRecognize.PlateDetector as PlateDetector
import PlateRecognize.CharsPredict as CharsPredict
import cv2

is_saveGray = False
det = PlateDetector.PlateDetector()
seg = PlateSegment.PlateSegment(det.getImageNormalizedWidth(), det.getImageNormalizedHeight())


# TODO(FesianXu) 对于识别失败的或者成功矫正的，需要返回其失败类型和矫正类型，以方便后续的实验纪录和改进
def recognizePlate(img):
    '''
    :: 对输入的图片进行识别
    :param img: 图片，彩色三通道
    :return: 识别的结果列表, 矫正类型list，车牌图片
    '''
    img_mat = det.getPlateRegion(img)
    img_out_bin, img_out_gray, correct_types = det.plateCorrect(img_mat)
    img_out_real_bin, img_out_real_gray, correct_real_types, res_list = [], [], [], []
    for ind, each in enumerate(img_out_bin):
        roi_set = seg.plateSegment(each, is_saveGray)
        if roi_set is not None:
            res_list_tmp = CharsPredict.predict_chars(roi_set)
            res_list.append(res_list_tmp)
            img_out_real_bin.append(img_out_bin[ind])
            img_out_real_gray.append(img_out_gray[ind])
            correct_real_types.append(correct_types[ind])
    return res_list, correct_types, img_out_bin, img_out_gray


########################################################################################################################

import time

if __name__ == '__main__':
    path = 'F:/opencvjpg/'
    name = '1014.jpg'
    file_name = path+name
    img = cv2.imread(file_name)
    while True:
        begin = time.clock()
        res = recognizePlate(img)
        end = time.clock()
        print(res)
        print('time cost = ', end-begin)
    pass