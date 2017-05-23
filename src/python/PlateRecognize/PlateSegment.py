# !/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'FesianXu'
__date__ = '2017/5/21'
__version__ = 'version 0.1'

'''
将矫正好的车牌进行字符分割
'''

import test
import cv2
import numpy as np
import matplotlib.pyplot as plt

class PlateSegment(object):
    '''
    用于分割已经矫正好了的车牌中的字符，以供下一步字符识别使用
    '''
    __divide_points = (0.086, 0.216, 0.395, 0.525, 0.654, 0.784, 0.913)  # 依靠着车牌字符的顺序排列的比例，左方向为初始方向
    __chars_wh_upper = 5
    __chars_wh_lower = 1.5  # 字符比例
    __chars_area_lower = 50  # 字符最小面积
    __chars_area_upper = 500  # 字符最大面积
    __char_width_lower = 10
    __char_height_lower = 10
    __char_width_upper = 40
    __char_height_upper = 40

    def __init__(self):
        pass


    def __drawCharBox(self, img, center, width, height):

        pass


    def __decideCharsTypes(self, real_contours):
        '''
        :: 求得目前能够知道的字符的对于车牌的相对位置
        :param real_contours: 真实字符轮廓
        :return: 相对位置
        '''
        centers_loc = []
        width_set = []
        height_set = []
        for each in real_contours:
            max_row, min_row, max_col, min_col = max(each[:, :, 1]), min(each[:, :, 1]), max(each[:, :, 0]), min(each[:, :, 0])
            width_set.append(max_col-min_col)
            height_set.append(max_row-min_row)
            loc = ((max_col+min_col)/2, (max_row+min_row)/2)
            centers_loc.append(loc)
        width_ideal = np.mean(width_set)
        height_ideal = np.mean(height_set)
        centers_loc = np.array(centers_loc)
        return centers_loc





    @test.timeit
    def __deleteSmallCharRegions(self, img):
        '''
        :: 除去图片中明显不是字符的区域
        :param img:
        :return: 真实的字符轮廓
        '''
        _, chars_contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        real_contours = []
        for each in chars_contours:
            dcol, drow = max(each[:, :, 0])-min(each[:, :, 0]), max(each[:, :, 1])-min(each[:, :, 1])
            area = cv2.contourArea(each)
            if self.__chars_wh_lower <= drow/dcol <= self.__chars_wh_upper and self.__chars_area_lower <= area <= self.__chars_area_upper:
                real_contours.append(each)
        return real_contours


    def plateSegment(self, img):
        '''
        :: 对二值车牌进行字符分割
        :param img: 车牌的二值图
        :return:
        '''
        chars_contours = self.__deleteSmallCharRegions(img)
        loc = self.__decideCharsTypes(chars_contours)
        plt.imshow(img)
        for each in chars_contours:
            plt.scatter(each[:,:,0], each[:,:,1], color='r')
            plt.scatter(loc[:,0], loc[:,1], color='b')
        plt.show()


########################################################################################################################

import PlateRecognize.PlateDetector as PlateDetector


if __name__ == '__main__':
    path = 'F:/opencvjpg/'
    name = '41.jpg'
    file_name = path+name
    img = cv2.imread(file_name)
    det = PlateDetector.PlateDetector()
    seg = PlateSegment()
    img_mat = det.getPlateRegion(img)
    img_out = det.plateCorrect(img_mat)
    for each in img_out:
        seg.plateSegment(each)
