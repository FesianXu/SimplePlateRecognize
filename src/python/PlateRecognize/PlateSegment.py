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
import line_profiler
import sys

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

    def __init__(self, img_width, img_height):
        self.__img_width = img_width
        self.__img_height = img_height
        pass


    def __drawCharBox(self, img, center, width, height):

        pass


    # @test.timeit
    def __decideCharsTypes(self, centers_loc):
        '''
        :: 求得目前能够知道的字符的对于车牌的相对位置, 可能会有重复的，因为内轮廓和外轮廓表示同一个字符
        :param: centers_loc 字符中心位置list
        :return: 分割了的相对位置和中心坐标的对应list， 和未分割的字符位置
        '''
        centers_loc = np.array(centers_loc).reshape(len(centers_loc), 2)
        proportion = centers_loc[:, 0]/self.__img_width
        type_list = []
        for each_centers in proportion:
            diff_list = []
            for each_divide in self.__divide_points:
                diff_list.append(abs(each_centers-each_divide))
            type_list.append(diff_list.index(min(diff_list)))
        return type_list


    def __mergeDuplicateContours(self, type_list, centers_loc):
        '''
        :: 融合重复的内轮廓和外轮廓中心位置
        :param type_list: 类型字段，可能有重复的需要融合
        :param centers_loc: 字符中心位置
        :return: 类型字段，中心位置配对list (type_list, center_loc_list)
        '''
        new_type_list, new_centers_list = [-1]*7, [-1]*7
        for each in type_list:
            sum_row, sum_col = 0, 0
            dup = [ind for ind, content in enumerate(type_list) if content == each]
            new_type_list[each] = each+1
            if len(dup) == 1:
                new_centers_list[each] = centers_loc[dup[0]]
            else:
                for each_dup in dup:
                    sum_row += centers_loc[each_dup][1]
                    sum_col += centers_loc[each_dup][0]
                avg_row, avg_col = sum_row/len(dup), sum_col/len(dup)
                new_centers_list[each] = (avg_col, avg_row)
        return new_type_list, new_centers_list


    def __cutTheChars(self, center_set, width_set, height_set):
        '''
        :: 切割车牌中的字符
        :param center_set: 中心位置集合
        :param width_set: 长度集合
        :param height_set: 宽度结合
        :return: 切割好的字符图片，保存为二值图或者灰度图（灰度图需要援引）
        '''
        pass

    def __cutTheChars(self, gray_img, center_set, width_set, height_set):
        pass


    def __getMissingCharsMsg(self):

        pass


    # @test.timeit
    def __getCharsBoxingMsg(self, real_contours):
        centers_loc = []
        width_set = []
        height_set = []
        for each in real_contours:
            max_row, min_row, max_col, min_col = max(each[:, :, 1]), min(each[:, :, 1]), max(each[:, :, 0]), min(each[:, :, 0])
            width_set.append(max_col-min_col)
            height_set.append(max_row-min_row)
            loc = ((max_col+min_col)/2, (max_row+min_row)/2)
            centers_loc.append(loc)
        width_ideal = np.max(width_set)
        height_ideal = np.max(height_set)
        return centers_loc, width_ideal, height_ideal


    def __deleteSmallCharRegions(self, img):
        '''
        :: 除去图片中明显不是字符的区域
        :param img:
        :return: 真实的字符轮廓
        '''
        _, chars_contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        real_contours = []
        for each in chars_contours:
            dcol, drow = max(each[:, :, 0])-min(each[:, :, 0]), max(each[:, :, 1])-min(each[:, :, 1])
            # dcol+0.0001 防止出现除0异常
            if self.__chars_wh_lower <= (drow/(dcol+0.0001)) <= self.__chars_wh_upper:
                area = cv2.contourArea(each)
                if self.__chars_area_lower <= area <= self.__chars_area_upper:
                    real_contours.append(each)
        return real_contours

    # @test.timeit
    def plateSegment(self, img):
        '''
        :: 对二值车牌进行字符分割
        :param img: 车牌的二值图
        :return:
        '''
        chars_contours = self.__deleteSmallCharRegions(img)
        boxmsg = self.__getCharsBoxingMsg(chars_contours)
        type_list = self.__decideCharsTypes(boxmsg[0])
        type_list, center_list = self.__mergeDuplicateContours(type_list, boxmsg[0])
        print(type_list)
        print(center_list)
        loc = np.array(boxmsg[0])
        plt.imshow(img)
        for each in chars_contours:
            plt.scatter(each[:,:,0], each[:,:,1], color='r')
            plt.scatter(loc[:,0], loc[:,1], color='b')
        plt.show()


########################################################################################################################

import PlateRecognize.PlateDetector as PlateDetector

path = 'F:/opencvjpg/'
name = '41.jpg'
file_name = path+name
img = cv2.imread(file_name)

def main():
    det = PlateDetector.PlateDetector()
    img_mat = det.getPlateRegion(img)
    img_out = det.plateCorrect(img_mat)
    seg = PlateSegment(det.getImageNormalizedWidth(), det.getImageNormalizedHeight())
    for each in img_out:
        seg.plateSegment(each)


if __name__ == '__main__':
    main()
