# !/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'FesianXu'
__date__ = '2017/6/1'
__version__ = 'version 0.1'

'''
自动运行脚本，对一个文件夹内所有文件进行处理
'''


import os
import time
import cv2
import PlateRecognize.recognizePlate as recog
import PlateRecognize.logger as logger

root_path = 'F:/opencvjpg/new_plate_img/'
save_plate_path = ''

lg = logger.PlateLogger(isRecord=True)

if __name__ == '__main__':
    dirlist = os.listdir(root_path)
    begin = time.clock()
    valid_num = 0
    invalid_list = []
    valid_list = []
    correct_types_list = []
    lg.log_info()
    lg.log_date()
    lg.log_time()
    lg.log_img_folder(root_path)
    lg.log_platform()
    for each in dirlist:
        file_name = root_path+each
        img = cv2.imread(file_name)
        try:
            res, correct_types = recog.recognizePlate(img)
            valid_num += 1
            print('process in = ', valid_num)
            valid_list.append(each)
            correct_types_list.append(correct_types)
        except Exception as e:
            invalid_list.append(each)
            print(e)
            continue
    end = time.clock()
    print('cost = ', end-begin)
    print('loop = ', valid_num)
    print('total = ', len(dirlist))
    lg.log_cost_time(begin, end, len(dirlist), valid_num)
    lg.log_valid_list(valid_list, correct_types_list)
    lg.log_invalid_list(invalid_list)
    lg.file_close()