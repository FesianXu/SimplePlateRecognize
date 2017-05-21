# !/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'FesianXu'
__date__ = '2017/5/21'
__version__ = 'version 0.1'

'''
PlateDetector, use to detect the plate frame location in an image
'''

import cv2
import numpy as np
import ImageManager
import test




class PlateDetector(object):
    '''
    检测图像中车牌的区域
    '''
    __img_levels = 255
    __h_img_upper = 0.720*180
    __h_img_lower = 0.555*180 # h只有[0,180]
    __s_img_upper = 1.0*__img_levels
    __s_img_lower = 0.400*__img_levels
    __v_img_upper = 1.000*__img_levels
    __v_img_lower = 0.300*__img_levels
    __plate_wh_upper = 5  # 车牌长宽比例上限
    __plate_wh_lower = 1.3  # 车牌长宽比例下限
    __plate_wh_least_width = 50  # 车牌至少长度
    __plate_wh_least_height = 30  # 车牌至少宽度

    def __init__(self):
        pass

    def __getBlueRegion(self, img):
        '''
        :param img: 原图像
        :return: 可能的蓝色区域掩膜，值范围归一化至[0,1]
        '''
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_h = img_hsv[:, :, 0]
        img_s = img_hsv[:, :, 1]
        img_v = img_hsv[:, :, 2]
        img_h_mask = (img_h <= self.__h_img_upper) & (img_h >= self.__h_img_lower)
        img_s_mask = (img_s <= self.__s_img_upper) & (img_s >= self.__s_img_lower)
        img_v_mask = (img_v <= self.__v_img_upper) & (img_v >= self.__v_img_lower)
        img_blue = (img_h_mask & img_s_mask & img_v_mask).astype(np.uint8)
        return img_blue

    @test.timeit
    def getPlateRegion(self, img):
        '''
        :param img: 输入图片
        :return: 输出可能的车牌区域
        '''
        img = ImageManager.normalizeImage(img)
        blue = self.__getBlueRegion(img)
        erode_kernel = np.ones((2, 2), np.uint8)
        dilate_kernel = np.ones((8, 18), np.uint8)
        blue = cv2.erode(blue, erode_kernel)
        blue = cv2.dilate(blue, dilate_kernel)
        image, contours, _ = cv2.findContours(blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        img_mat = []
        for each in contours:
            dcol, drow = max(each[:, :, 0])-min(each[:, :, 0]), max(each[:, :, 1])-min(each[:, :, 1])
            if self.__plate_wh_lower <= dcol/(drow+0.0001) <= self.__plate_wh_upper and dcol >= self.__plate_wh_least_width and drow >= self.__plate_wh_least_height:
                max_row, min_row = max(each[:, :, 1])[0], min(each[:, :, 1])[0]
                max_col, min_col = max(each[:, :, 0])[0], min(each[:, :, 0])[0]
                img_tmp = img[min_row:max_row, min_col:max_col, :]
                img_tmp = ImageManager.normalizePlateRegion(img_tmp)
                img_mat.append(img_tmp)
        return img_mat

    def __rotateAngle(self, img):
        '''
        :param img: 未校准车牌区域
        :return: 倾斜角度
        '''
        pass


    @test.timeit
    def plateCorrect(self, img_mat):
        for img in img_mat:
            img_blue = self.__getBlueRegion(img)
            _, contours, _ = cv2.findContours(img_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            plate_list = []
            for each in contours:
                dcol, drow = max(each[:, :, 0])-min(each[:, :, 0]), max(each[:, :, 1])-min(each[:, :, 1])
                if self.__plate_wh_lower <= dcol/(drow+0.0001) <= self.__plate_wh_upper and dcol >= self.__plate_wh_least_width \
                        and drow >= self.__plate_wh_least_height:
                    plate_list.append(each)
            if plate_list:
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _, img_bin = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU)
                # angle = self.__rotateAngle(img_gray)
                show(img_bin)
                pass





########################################################################################################################

def show(img):
    cv2.imshow('w', img)
    cv2.waitKey(-1)

@test.timeit
def main():

    path = 'F:/opencvjpg/'
    name = '1016.jpg'
    file_name = path+name
    img = cv2.imread(file_name)
    det = PlateDetector()
    blue = det.getPlateRegion(img)

    det.plateCorrect(blue)

if __name__ == '__main__':
    cv2.setUseOptimized(True)
    main()


