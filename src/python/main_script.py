# !/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'FesianXu'
__date__ = '2017/5/23'
__version__ = 'version 0.1'

'''
main脚本，用于运行主程序，主要进行一系列新算法的尝试
'''

import PlateRecognize.PlateSegment as PlateSegment
import PlateRecognize.PlateDetector as PlateDetector
# import PlateRecognize.recognizePlate as recog
import matplotlib.pyplot as plt
import cv2


det = PlateDetector.PlateDetector()
seg = PlateSegment.PlateSegment(det.getImageNormalizedWidth(), det.getImageNormalizedHeight())
is_saveGray = False

if __name__ == '__main__':
    path = 'F:/opencvjpg/new_plate_img/'
    name = '1014.jpg'
    file_name = path+name
    img = cv2.imread(file_name)
    # res, correct_types, img_bin, _ = recog.recognizePlate(img)
    # print(res)

    img_mat = det.getPlateRegion(img)
    img_out_bin, img_out_gray, _ = det.plateCorrect(img_mat)
    for ind, each in enumerate(img_out_bin):
        roi_set = seg.plateSegment(each, is_saveGray)
        if roi_set is not None:
            for ind_i, each_i in enumerate(roi_set):
                plt.subplot(1, 7, ind_i+1)
                plt.imshow(each_i, cmap='gray')
            plt.show()
