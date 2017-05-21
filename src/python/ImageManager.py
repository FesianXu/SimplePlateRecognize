# !/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'FesianXu'
__date__ = '2017/5/20'
__version__ = 'version 0.1'

'''
Class ImageCapture , to capture the image from video or folder
Class ImagePreProcess
'''

import cv2
import os
import random
import test


class ImageCapture(object):

    def __init__(self, path, isCap = False):
        '''
        :param folder_path: 读取图片文件夹目录
        :return: None
        '''
        self.__file_path = path
        pass


    def getImages(self, size=-1):
        file_dir = os.listdir(self.__file_path)
        img_count = len(file_dir)
        img_mat = []
        if size < 0 or size > img_count:
            for each in file_dir:
                file_name = self.__file_path+each
                img = cv2.imread(file_name)
                img_mat.append(img)
        elif size >= 0 and size <= img_count:
            img_rand = random.sample(file_dir, size)
            for each in img_rand:
                file_name = self.__file_path+each
                img = cv2.imread(file_name)
                img_mat.append(img)
        return img_mat

########################################################################################################################


def turnBinaryMat(img_mat, max_val=1, thresh=120):
    '''
    :param img_mat: 输入图像列表
    :param max_val: 最大bin值
    :param thresh:  二值化阀值
    :return: 二值化后的图像列表
    '''
    bin_mat = []
    for each in img_mat:
        if each.ndim == 3:
            each = cv2.cvtColor(each, cv2.COLOR_BGR2GRAY)
        _, bin_img = cv2.threshold(each, thresh, max_val, cv2.THRESH_BINARY)
        img = bin_img.reshape([1, each.size])
        bin_mat.append(img)
    return bin_mat


def normalizeImage(img,width=800, height=600):
    return cv2.resize(img, (width, height))


def normalizePlate(img, width=180, height=50):
    return cv2.resize(img, (width, height))


def normalizePlateRegion(img, width=300, height=140):
    return cv2.resize(img, (width, height))


def gaussian(img, core=(5, 5), sigmaX=2, sigmaY=2):
    return cv2.GaussianBlur(img, core, sigmaX, sigmaY)


if __name__ == '__main__':
    path = '../../res/trainning_set/normalized/numbers/9/'
    imgtest = ImageCapture(path)

