# !/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'FesianXu'
__date__ = '2017/5/26'
__version__ = ''

'''
得到一些车牌区域和非车牌区域做SVM训练
'''

import os
import cv2
import PlateRecognize.PlateDetector as PlateDetector
import time

image_path = 'F:/opencvjpg/new_plate_img/'

save_path = 'F:/opencvjpg/training sets/plates/'

det = PlateDetector.PlateDetector()

def main(each):
    file_name = image_path+each
    img = cv2.imread(file_name)

    img_mat = det.getPlateRegion(img)
    img_out_bin, img_out_gray = det.plateCorrect(img_mat)
    return img_out_bin

if __name__ == '__main__':
    dir_list = os.listdir(image_path)
    begin = time.clock()
    for ind, each in enumerate(dir_list):
        print(ind)
        try:
            imgout = main(each)
            for ind_i, each_img in enumerate(imgout):
                save_name = save_path+str(ind)+'--'+each+'.png'
                cv2.imwrite(save_name, each_img)
                # cv2.imshow('tt', each_img)
                # cv2.waitKey(-1)
        except Exception:
            continue
    end = time.clock()
    print('time cost = ', end-begin)

    # 325 => 67.299s 303 items
    # 325 => 74.442
