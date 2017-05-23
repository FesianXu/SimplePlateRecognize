# !/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'FesianXu'
__date__ = '2017/5/23'
__version__ = 'version 0.1'

'''
main脚本，用于运行主程序
'''

import PlateRecognize.PlateDetector as PlateDetector
import cv2

if __name__ == '__main__':
    detector = PlateDetector.PlateDetector()
    path = 'F:/opencvjpg/'
    name = '1014.jpg'
    file_name = path+name
    img = cv2.imread(file_name)
    img_mat = detector.getPlateRegion(img)
    imgout = detector.plateCorrect(img_mat)
    for each in imgout:
        cv2.imshow('w', each)
        cv2.waitKey(-1)
