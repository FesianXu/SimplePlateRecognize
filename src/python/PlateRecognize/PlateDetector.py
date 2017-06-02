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
import os
import math

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from svmutil import *


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
    __plate_norm_width = 180  # 车牌标准化长度
    __plate_norm_height = 50  # 车牌标准宽度
    __img_norm_width = 800  # 图像标准化长度
    __img_norm_height = 600  # 图像标准化宽度
    __plate_region_norm_width = 300  # 车牌区域未校准时的标准化长度
    __plate_region_norm_height = 140  # 车牌区域未校准时的标准化宽度
    __plate_tilt_type1 = 3  # 以下的可以不进行倾斜矫正，无风险
    __plate_tilt_type2 = 7  # 以下的进行简单的旋转矫正， 风险较小，主要集中在测量角度的精确度上
    # 其余的进行透视变换矫正，风险最大，主要集中在四点定位准确度上
    __is_plate_model_save_path = '/src/python/train_data/is_plate/is_plate_svm_model.model'
    __project_root_path = u''  # 绝对项目路径
    __isSet = False  # 是否已经设置了绝对项目路径
    __is_plate_svm_model = None  # 判断是否是车牌的svm模型


    def __init__(self):
        '''
        :: 加载SVM模型，用于判断是否是车牌
        '''
        if self.__isSet is not True:
            BASE_DIR = self.__project_root_path = os.path.dirname(__file__)
            self.__project_root_path = os.path.dirname(os.path.dirname(os.path.dirname(BASE_DIR)))
            self.__is_plate_model_save_path = self.__project_root_path+self.__is_plate_model_save_path
            self.__is_plate_svm_model = svm_load_model(self.__is_plate_model_save_path)
        self.__isSet = True


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


    def getPlateRegion(self, img):
        '''
        :: 得出车牌的区域，未经过矫正，可能还是倾斜的
        :param img: 输入图片
        :return: 输出可能的车牌区域
        '''
        img = cv2.resize(img, (self.__img_norm_width, self.__img_norm_height))
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
                img_tmp = cv2.resize(img_tmp, (self.__plate_region_norm_width, self.__plate_region_norm_height))
                img_mat.append(img_tmp)
        return img_mat


    def __imrotate(self, img, angle, scale=0.75):
        width = img.shape[1]
        height = img.shape[0]
        rotate_mat = cv2.getRotationMatrix2D((width/2, height/2), angle, scale)
        rotate_img = cv2.warpAffine(img, rotate_mat, (width, height))
        return rotate_img


    # TODO(FesianXu) 在得到k的过程中，除以sin(theta)容易出现除0异常
    def __rotateAngle(self, img):
        '''
        :: 求车牌的倾斜角，以决定矫正方案
        :param img: 未校准车牌区域, 二值图
        :return: 倾斜角度
        '''
        img_canny = cv2.Canny(img*255, 50, 150)
        lines = cv2.HoughLines(img_canny, 1, np.pi/180, threshold=80)
        angle_list = []
        if lines is not None:
            for each in lines:
                theta = each[0, 1]
                k = -np.cos(theta)/np.sin(theta)
                angle = round(np.arctan(k)*180/np.pi)
                angle_list.append(angle)
        else:
            return None
        angle_avg = sum(angle_list)/len(angle_list)
        return angle_avg


    # @test.timeit
    # TODO(FesianXu) 这里的kmeans可以考虑优化为简单聚类即可
    def __getQuadrangleVertices(self, img, hull):
        '''
        :: 得到倾斜车牌的四个顶点，以用于求得单因性矩阵
        :param img: 车牌彩色图像
        :param hull: 车牌凸包
        :return: 车牌的四个顶点，list储存
        '''
        cosine_thresh_upper, cosine_thresh_lower = np.cos(30*np.pi/180), np.cos(155*np.pi/180)
        calc_step = 4
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
            tmpcos.append([angle, pmedian])
        for eachcos, point in tmpcos:
            if cosine_thresh_lower <= eachcos <= cosine_thresh_upper:
                vertices_list.append(point)
        vertices_list = np.array(vertices_list)
        vertices_list_reshape = vertices_list.reshape((len(vertices_list), 2))
        # 这里的cluster有可能因为边角点数不足而错误，需要异常捕获处理。
        k_cluster = KMeans(n_clusters=4, random_state=0).fit(vertices_list_reshape)
        centers = sorted(k_cluster.cluster_centers_.tolist(), key=lambda x: x[0])  # sorted by col
        vertices_left, vertices_right = centers[0:2], centers[2:4]
        vertices_left = sorted(vertices_left, key=lambda x: x[1])
        vertices_right = sorted(vertices_right, key=lambda x: x[1])
        p1, p2, p3, p4 = vertices_left[0], vertices_right[0], vertices_left[1], vertices_right[1]
        vertices = [p1, p2, p3, p4]
        # vertices = np.array(vertices)
        # plt.imshow(img[:,:,::-1])
        # plt.scatter(vertices_list[:, 0, 0], vertices_list[:, 0, 1], color='r')
        # plt.scatter(vertices[:, 0], vertices[:, 1], color='b')
        # plt.show()
        return vertices


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
            return img_correct
        else:
            return None


    # @test.timeit
    def __isPlate(self, img):
        '''
        :: 判断是否是车牌，利用SVM判断
        :: 实际应用中因为采用的分类方法不好，效果不好 accuracy=62%
        :param img: 待测图像二值图
        :return:
        '''
        f = img.reshape(1, img.size).astype(float).tolist()
        y = [1]
        label, acc, val = svm_predict(y, f, self.__is_plate_svm_model)
        if label[0] == 1.0:
            return True
        else:
            return False


    # @test.timeit
    def __deletePlateFrames(self, img, thresh=10):
        '''
        :: 删除矫正后车牌的边框，主要根据的是边缘跳变信息
        :param img: 车牌二值图
        :param thresh:
        :return:
        '''
        row = img.shape[0]
        img_out = np.zeros(img.shape, np.uint8)
        for eachrow in range(row):
            horlist = img[eachrow, :]
            horlist_tmp = np.concatenate(([0], img[eachrow, :]), axis=0)
            horlist = np.concatenate((horlist, [0]), axis=0)
            delta = np.absolute((horlist-horlist_tmp))
            delta_sum = np.sum(delta, dtype=np.int32)
            if delta_sum >= thresh*255:
                img_out[eachrow, :] = img[eachrow, :]
        return img_out

    # import test
    #
    # @test.timeit
    # TODO(FesianXu) 需要倾斜角度检测，以判断用何种方法做倾斜矫正。(complete 2017/6/2)
    # TODO(Fesianxu) 需要添加旋转矫正车牌的选项。(complete 2017/6/2)
    # TODO(Fesianxu) 加入SVM判断是否是车牌 (completed 2017/5/28)
    # TODO(FesianXu) 优化旋转矫正
    def plateCorrect(self, img_mat):
        '''
        :: 矫正车牌，利用三种可能的方法矫正 判断是否是车牌，通过SVM
        :param img_mat: 未矫正的车牌区域集合
        :return: 判断后，并且矫正过后的车牌集合，数量不一定等于img_mat, 返回矫正类型
        '''
        img_out_bin = []
        img_out_gray = []
        correct_types = []
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
                angle_avg = self.__rotateAngle(img_blue)  # 得到车牌的倾斜程度
                # print(angle_avg)
                if angle_avg is None:
                    continue
                plate_list_valid = plate_list[0]
                if abs(angle_avg) <= self.__plate_tilt_type1:  # 不进行矫正，只是标准化图像
                    max_col, min_col = max(plate_list_valid[:, :, 0])[0], min(plate_list_valid[:, :, 0])[0]
                    max_row, min_row = max(plate_list_valid[:, :, 1])[0], min(plate_list_valid[:, :, 1])[0]
                    img_correct = img[min_row:max_row, min_col:max_col, :]
                    img_correct = cv2.resize(img_correct, (self.__plate_norm_width, self.__plate_norm_height))
                    correct_types.append(0)
                elif self.__plate_tilt_type1 < abs(angle_avg) < self.__plate_tilt_type2:  # 进行简单旋转矫正
                    img_correct = self.__imrotate(img, angle_avg)
                    img_correct_blue = self.__getBlueRegion(img_correct)
                    dilate_kernel = np.ones((10, 15), np.uint8)
                    img_correct_blue = cv2.dilate(img_correct_blue, dilate_kernel)
                    _, img_correct_contours, _ = cv2.findContours(img_correct_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    img_correct_list = []
                    for each_blue in img_correct_contours:
                        dcol, drow = max(each_blue[:, :, 0])-min(each_blue[:, :, 0]), max(each_blue[:, :, 1])-min(each_blue[:, :, 1])
                        if self.__plate_wh_lower <= dcol/(drow+0.0001) <= self.__plate_wh_upper and dcol >= self.__plate_wh_least_width \
                                and drow >= self.__plate_wh_least_height:
                            img_correct_list.append(each_blue)
                    valid_blue = img_correct_list[0]
                    max_col, min_col = max(valid_blue[:, :, 0])[0], min(valid_blue[:, :, 0])[0]
                    max_row, min_row = max(valid_blue[:, :, 1])[0], min(valid_blue[:, :, 1])[0]
                    img_correct = img_correct[min_row:max_row, min_col:max_col, :]
                    img_correct = cv2.resize(img_correct, (self.__plate_norm_width, self.__plate_norm_height))
                    correct_types.append(1)
                else:
                    img_correct = self.__projectionCorrect(img, plate_list_valid)
                    correct_types.append(2)

                img_correct_gray = cv2.cvtColor(img_correct, cv2.COLOR_BGR2GRAY)
                _, img_correct = cv2.threshold(img_correct_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                if self.__isPlate(img_correct):  # 通过svm，判断是否是车牌
                    img_frame = self.__deletePlateFrames(img_correct)
                    # show(img_frame)
                    erode_kernel = np.ones((2, 2))
                    img_frame = cv2.erode(img_frame, erode_kernel)
                    img_out_bin.append(img_frame)
                    img_out_gray.append(img_correct_gray)
        return img_out_bin, img_out_gray, correct_types

    def getImageNormalizedWidth(self):
        return self.__plate_norm_width

    def getImageNormalizedHeight(self):
        return self.__plate_norm_height




########################################################################################################################

def show(img):
    cv2.imshow('w', img)
    cv2.waitKey(-1)


# @test.timeit
def main():

    path = 'F:/opencvjpg/'
    name = '41.jpg'
    file_name = path+name
    img = cv2.imread(file_name)
    det = PlateDetector()
    blue = det.getPlateRegion(img)
    plates = det.plateCorrect(blue)
    for each in plates:
        show(each)

if __name__ == '__main__':
    cv2.setUseOptimized(True)
    main()



