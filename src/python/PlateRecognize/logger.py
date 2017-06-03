# !/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'FesianXu'
__date__ = '2017/5/23'
__version__ = 'version 0.1'
__platform__ = 'Python 3.5 in Anaconda, with TensorFlow, LibSVM, scikit-learn, opencv-python'

'''
用于实验纪录文档生成
'''
import os
# import platform
import time
import cv2

class PlateLogger(object):

    __log_record_path = '/test_record/txt_files/'  # 实验记录文件夹
    __log_save_result_plate_path = 'F:/opencvjpg/training sets/plates/'
    __isSet = False  # 是否已经设置了绝对项目路径
    __project_root_path = ''
    __file_handle = None
    __isRecord = True
    __isSaveResultImages = True

    def __init__(self, isRecord=True, isSaveResultImages=True):
        self.__isRecord = isRecord
        self.__isSaveResultImages = isSaveResultImages
        if self.__isRecord is True:
            if self.__isSet is False:
                self.__isSet = True
                BASE_DIR = self.__project_root_path = os.path.dirname(__file__)
                self.__project_root_path = os.path.dirname(os.path.dirname(os.path.dirname(BASE_DIR)))
                self.__log_record_path = self.__project_root_path+self.__log_record_path
                current_time = time.strftime("20%y-%m-%d, %H-%M-%S")
                file_name = self.__log_record_path+current_time+'.txt'
                file_name = file_name
                self.__file_handle = open(file_name, 'w')

    def log_invalid_list(self, name_list, except_list=None):
        if self.__isRecord is True:
            self.__file_handle.write('-----------------------------------------------------------------------------\r\n')
            self.__file_handle.write('∏  The name list of failing plates recognition: \r\n')
            if except_list is None:
                self.__file_handle.write('No.         '+"image's Name    \r\n")
                for ind, each in enumerate(name_list):
                    self.__file_handle.write('No.'+str(ind)+'            '+each+'   \r\n')
            else:
                self.__file_handle.write('No.           '+"image's Name     "+'Fail reasons    \r\n')
                for ind, each in enumerate(name_list):
                    self.__file_handle.write('No.'+str(ind)+'            '+each)
                    self.__file_handle.write('      '+str(except_list[ind])+'    \r\n')
            self.__file_handle.write('-----------------------------------------------------------------------------\r\n')


    def log_valid_list(self, name_list, correct_type_list=None):
        if self.__isRecord is True:
            self.__file_handle.write('-----------------------------------------------------------------------------\r\n')
            self.__file_handle.write('∏  The name list of successful plates recognition: \r\n')
            if correct_type_list is None:
                for ind, each in enumerate(name_list):
                    self.__file_handle.write('No.'+str(ind)+'            '+each+'  \r\n')
            else:
                type1_list, type2_list, type3_list = [], [], []
                type_list_all = [type1_list, type2_list, type3_list]
                for ind, each_type in enumerate(correct_type_list):
                    for types in [0, 1, 2]:
                        if types in each_type:
                            type_list_all[types].append(name_list[ind])
                self.__file_handle.write('***************************\r\n')
                self.__file_handle.write('Type 1 has %d images \r\n' % len(type1_list))
                self.__file_handle.write('Type 2 has %d images \r\n' % len(type2_list))
                self.__file_handle.write('Type 3 has %d images \r\n' % len(type3_list))
                self.__file_handle.write('***************************\r\n')
                self.__file_handle.write('∮ Don not need to correct images name list: \r\n')
                for ind, each in enumerate(type1_list):
                    self.__file_handle.write('No.'+str(ind+1)+'   '+each+'\r\n')
                self.__file_handle.write('***************************\r\n')
                self.__file_handle.write('∮ Just simply rotate the plate to correct images name list: \r\n')
                for ind, each in enumerate(type2_list):
                    self.__file_handle.write('No.'+str(ind+1)+'   '+each+'\r\n')
                self.__file_handle.write('***************************\r\n')
                self.__file_handle.write('∮ Use projective transform to correct plate images name list: \r\n')
                for ind, each in enumerate(type3_list):
                    self.__file_handle.write('No.'+str(ind+1)+'   '+each+'\r\n')
                self.__file_handle.write('***************************\r\n')
            self.__file_handle.write('-----------------------------------------------------------------------------\r\n')


    def log_platform(self):
        if self.__isRecord is True:
            self.__file_handle.write('-----------------------------------------------------------------------------\r\n')
            self.__file_handle.write('∏  Run in platform = %s \r\n' % __platform__)
            self.__file_handle.write('-----------------------------------------------------------------------------\r\n')


    def log_time(self):
        if self.__isRecord is True:
            self.__file_handle.write('-----------------------------------------------------------------------------\r\n')
            self.__file_handle.write('∏  current time = '+time.strftime("%H:%M:%S  \r\n"))
            self.__file_handle.write('-----------------------------------------------------------------------------\r\n')


    def log_date(self):
        if self.__isRecord is True:
            self.__file_handle.write('-----------------------------------------------------------------------------\r\n')
            self.__file_handle.write('∏  current date = '+time.strftime("20%y-%m-%d \r\n"))
            self.__file_handle.write('-----------------------------------------------------------------------------\r\n')


    def log_img_folder(self, folder_path):
        if self.__isRecord is True:
            self.__file_handle.write('-----------------------------------------------------------------------------\r\n')
            self.__file_handle.write('∏  Recognize the plates in folder = %s \r\n' % folder_path)
            self.__file_handle.write('-----------------------------------------------------------------------------\r\n')

    def log_info(self):
        if self.__isRecord is True:
            self.__file_handle.write('-----------------------------------------------------------------------------\r\n')
            self.__file_handle.write('Author: %s \r\n' % __author__)
            self.__file_handle.write('Version: %s \r\n' % __version__)
            self.__file_handle.write('Coding Date: %s \r\n' % __date__)
            self.__file_handle.write('-----------------------------------------------------------------------------\r\n')

    def log_cost_time(self, begin, end, size_imgs, valid_num):
        if self.__isRecord is True:
            self.__file_handle.write('-----------------------------------------------------------------------------\r\n')
            self.__file_handle.write('∏    The costing time = %f seconds, with processing %d images \r\n' % (end-begin, size_imgs))
            self.__file_handle.write('      Each images takes %f seconds average \r\n' % ((end-begin)/size_imgs))
            self.__file_handle.write('      Valid images takes %f seconds average \r\n' % ((end-begin)/valid_num))
            self.__file_handle.write('      Valid images counts = %d \r\n' % valid_num)
            self.__file_handle.write('      Invalid images counts = %d \r\n' % (size_imgs-valid_num))
            self.__file_handle.write('      Total Image counts = %d \r\n' % size_imgs)
            self.__file_handle.write('      Valid Proportion: %f \r\n' % (valid_num/size_imgs))
            self.__file_handle.write('-----------------------------------------------------------------------------\r\n')


    def log_save_plate(self, img, res_list, image_raw_name=None):
        if self.__isSaveResultImages:
            if image_raw_name is None:
                name =('').join(res_list)+'.png'
            else:
                name = image_raw_name+'   '+('').join(res_list)+'.png'
                # 原图片名字  识别结果
            cv2.imencode('.png', img)[1].tofile(self.__log_save_result_plate_path+name)

    def log_loggerSetting(self):
        self.__file_handle.write('-----------------------------------------------------------------------------\r\n')
        self.__file_handle.write('∏    The logger setting is: \r\n')
        self.__file_handle.write('      The Logger files restore in folder: %s  \r\n' % self.__log_record_path)
        self.__file_handle.write('      Whether turn on recording function: %s \r\n' % self.__isRecord)
        self.__file_handle.write('      Whether turn on saving plates recognition result images: %s \r\n' % self.__isSaveResultImages)
        self.__file_handle.write('-----------------------------------------------------------------------------\r\n')


    def file_close(self):
        if self.__isRecord is True:
            self.__file_handle.close()

########################################################################################################################

if __name__ == '__main__':
    lg = PlateLogger()

    lg.log_info()
    lg.log_time()
    lg.log_date()
    lg.log_platform()
    lg.log_img_folder('C:/')
    lg.file_close()