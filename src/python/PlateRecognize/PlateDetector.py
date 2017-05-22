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
import matplotlib.pyplot as plt




class PlateDetector(object):
    '''
    检测图像中车牌的区域, 并予以车牌矫正， 边框去除，车牌区域与否判断等, 最终应得到一个经过矫正的合适的车牌区域，交于下一
    步字符分割
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
    __plate_norm_width = 160  # 车牌标准化长度
    __plate_norm_height = 48  # 车牌标准宽度

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
        :: 得出车牌的区域，未经过矫正，可能还是倾斜的
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

    @test.timeit
    def __rotateAngle(self, img):
        '''
        :: 求车牌的倾斜角，以决定矫正方案
        :param img: 未校准车牌区域, 二值图
        :return: 倾斜角度
        '''
        img_canny = cv2.Canny(img*255, 50, 150)
        lines = cv2.HoughLinesP(img_canny, 1, np.pi/180, threshold=100)

        if lines is not None:
            for each in lines:
                theta = each[0, 1]
                k = -np.cos(theta)/np.sin(theta)
                angle = round(np.arctan(k)*180/np.pi)
                print(angle)
        else:
            return None
        return 0

    def __getQuadrangleVertices(self, img, hull):
        '''
        :: 得到倾斜车牌的四个顶点，以用于求得单因性矩阵
        :param img: 车牌彩色图像
        :param hull: 车牌凸包
        :return: 车牌的四个顶点，list储存
        '''
        vertices = []
        cosine_thresh_upper, cosine_thresh_lower = 0.70, -0.70
        calc_step = 10
        sample_step = 1
        list_cosine = np.concatenate((hull, hull[0:calc_step, :, :]), axis=0)
        list_cosine = list_cosine[0::sample_step, :, :]
        tmpcos = []
        vertices_list = []
        for each in range(len(list_cosine)-calc_step):
            p1, p2 = list_cosine[each, :, :], list_cosine[each+calc_step, :, :]
            pmedian = list_cosine[each+int(calc_step/2), :, :]
            v1, v2 = np.squeeze(np.asarray(p1-pmedian)), np.squeeze(np.asarray(p2-pmedian))
            angle = np.dot(v1, v2.transpose())/(np.sqrt(v1.dot(v1))*np.sqrt(v2.dot(v2)))
            print(angle)
            tmpcos.append(angle)
        loop = 0
        for eachcos in tmpcos:
            if cosine_thresh_lower <= eachcos <= cosine_thresh_upper:
                vertices_list.append(list_cosine[loop, :, :])
            loop += 1
        vertices_list = np.array(vertices_list)



        plt.imshow(img[:,:,::-1])
        plt.plot(vertices_list[:, 0, 0], vertices_list[:, 0, 1], color='r')
        # plt.plot(hull[:,0,0], hull[:,0,1], color='b')
        plt.show()
        return vertices_list


    @test.timeit
    def __projectionCorrect(self, img, hull):
        '''
        :param img: 输入的彩色图像， 车牌区域，未校准
        :param hull: 车牌区域的凸包
        :return: 透视变换之后的车牌
        '''
        src_coordinate = self.__getQuadrangleVertices(img, hull)
        src_coordinate = np.array(src_coordinate, dtype=np.float32)
        dst_coordinate = [[0, 0], [self.__plate_norm_width, 0], [0, self.__plate_norm_height], [self.__plate_norm_width, self.__plate_norm_height]]
        dst_coordinate = np.array(dst_coordinate, dtype=np.float32)
        homograghy = cv2.getPerspectiveTransform(src_coordinate, dst_coordinate)
        if homograghy is not None:
            img_correct = cv2.warpPerspective(img, homograghy, (self.__plate_norm_width, self.__plate_norm_height))
            show(img_correct)


    @test.timeit
    def plateCorrect(self, img_mat):
        for img in img_mat:
            img_blue = self.__getBlueRegion(img)
            dilate_core = np.ones((10, 15), np.uint8)
            img_blue = cv2.dilate(img_blue, dilate_core)
            _, contours, _ = cv2.findContours(img_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            plate_list = []
            for each in contours:
                dcol, drow = max(each[:, :, 0])-min(each[:, :, 0]), max(each[:, :, 1])-min(each[:, :, 1])
                if self.__plate_wh_lower <= dcol/(drow+0.0001) <= self.__plate_wh_upper and dcol >= self.__plate_wh_least_width \
                        and drow >= self.__plate_wh_least_height:
                    hull = cv2.convexHull(each)
                    plate_list.append(hull)
            if plate_list:
                # 如果存在车牌
                # angle = self.__rotateAngle(img_blue)
                self.__projectionCorrect(img, plate_list[0])
                pass

    def deletePlateFrames(self, img, thresh):
        '''
        :: 删除矫正后车牌的边框，主要根据的是边缘跳变信息
        :param img:
        :param thresh:
        :return:
        '''
        pass



########################################################################################################################

def show(img):
    cv2.imshow('w', img)
    cv2.waitKey(-1)

@test.timeit
def main():

    path = 'F:/opencvjpg/'
    name = '41.jpg'
    file_name = path+name
    img = cv2.imread(file_name)
    det = PlateDetector()
    blue = det.getPlateRegion(img)

    det.plateCorrect(blue)

if __name__ == '__main__':
    cv2.setUseOptimized(True)
    main()



