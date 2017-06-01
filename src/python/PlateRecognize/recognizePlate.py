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

def recognizePlate(img):
    img_mat = det.getPlateRegion(img)
    img_out_bin, img_out_gray = det.plateCorrect(img_mat)
    for ind, each in enumerate(img_out_bin):
        roi_set = seg.plateSegment(each, is_saveGray)
        res = CharsPredict.predict_chars(roi_set)
    return res


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