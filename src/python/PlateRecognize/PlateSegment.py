# !/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'FesianXu'
__date__ = '2017/5/21'
__version__ = 'version 0.1'

'''
将矫正好的车牌进行字符分割
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class PlateSegment(object):
    '''
    用于分割已经矫正好了的车牌中的字符，以供下一步字符识别使用
    '''
    __divide_points = (0.086, 0.216, 0.395, 0.525, 0.654, 0.784, 0.913)  # 依靠着车牌字符的顺序排列的比例，左方向为初始方向
    __img_width = 0  # init in __init__
    __img_height = 0  # init in __init__
    __dist_between_chars = (0, 0)  # init in __init__
    __chars_normalized_width = 32  # 标准化后的字符长度
    __chars_normalized_height = 64  # 标准化后的字符宽度
    __chars_wh_upper = 5
    __chars_wh_lower = 1.5  # 字符比例
    __chars_area_lower = 50  # 字符最小面积
    __chars_area_upper = 500  # 字符最大面积
    __char_width_lower = 10
    __char_height_lower = 10
    __char_width_upper = 40
    __char_height_upper = 40
    __default_empty_centers = -10  # 预设的车牌中心数值，当未获取中心信息时候填充，需要小于-10

    def __init__(self, img_width, img_height):
        self.__img_width = img_width
        self.__img_height = img_height
        self.__dist_between_chars = (int(0.1795*self.__img_width), int(0.1295*self.__img_width))
        pass


    def __drawCharBox(self, img, center, width, height):

        pass


    # @test.timeit
    def __decideCharsTypes(self, centers_loc):
        '''
        :: 求得目前能够知道的字符的对于车牌的相对位置, 可能会有重复的，因为内轮廓和外轮廓表示同一个字符
        :param: centers_loc 字符中心位置list
        :return: 分割了的相对位置和中心坐标的对应list， 和未分割的字符位置
        '''
        centers_loc = np.array(centers_loc).reshape(len(centers_loc), 2)
        proportion = centers_loc[:, 0]/self.__img_width
        type_list = []
        for each_centers in proportion:
            diff_list = []
            for each_divide in self.__divide_points:
                diff_list.append(abs(each_centers-each_divide))
            type_list.append(diff_list.index(min(diff_list)))
        return type_list


    def __mergeDuplicateContours(self, type_list, centers_loc):
        '''
        :: 融合重复的内轮廓和外轮廓中心位置
        :param type_list: 类型字段，可能有重复的需要融合
        :param centers_loc: 字符中心位置
        :return: 类型字段，中心位置配对list (type_list, center_loc_list)
        '''
        new_type_list, new_centers_list = [self.__default_empty_centers]*7, [self.__default_empty_centers]*7
        for each in type_list:
            sum_row, sum_col = 0, 0
            dup = [ind for ind, content in enumerate(type_list) if content == each]
            new_type_list[each] = each+1
            if len(dup) == 1:
                new_centers_list[each] = centers_loc[dup[0]]
            else:
                for each_dup in dup:
                    sum_row += centers_loc[each_dup][1]
                    sum_col += centers_loc[each_dup][0]
                avg_row, avg_col = sum_row/len(dup), sum_col/len(dup)
                new_centers_list[each] = (avg_col, avg_row)
        return new_type_list, new_centers_list


    def __cutTheChars(self, plate_img, center_set, width_set, height_set, isgray=False):
        '''
        :: 切割车牌中的字符
        :param plate_img: 车牌图像，可以是二值图或者灰度图
        :param center_set: 中心位置集合
        :param width_set: 长度集合
        :param height_set: 宽度结合
        :param isgray: 是否需要保存为灰度图，前提是plate_img是灰度的
        :return: 切割好的字符图片，保存为二值图或者灰度图（灰度图需要指明，默认是二值图）
        '''
        roi_set = []
        for ind, each in enumerate(center_set):
            height_each, width_each = height_set[ind], width_set[ind]
            min_row, max_row = int(each[1]-height_each/2), int(each[1]+height_each/2)
            min_col, max_col = int(each[0]-width_each/2), int(each[0]+width_each/2)
            roi = plate_img[min_row:max_row, min_col:max_col]
            roi = cv2.resize(roi, (self.__chars_normalized_width, self.__chars_normalized_height))
            if isgray:
                pass
            else:
                _, roi = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)  # 二值图分割才需要二值化
            roi_set.append(roi)
        return roi_set


    # @test.timeit
    def __getMissingCharsMsgRoughly(self, type_list, center_list):
        '''
        :: 粗糙得到所有车牌字符中心的位置坐标
        :param type_list: 已经分割好的字符类型，从[0-6]
        :param center_list: 已经分割好的字符中心坐标。未获取的用__default_empty_centers表示
        :return: 所有字符的分割好的中心位置list
        '''
        missing_type = [ind+1 for ind, content in enumerate(type_list) if content == self.__default_empty_centers]
        missing_type = np.array(missing_type, np.int8)
        type_list = np.array(type_list, np.int8)
        new_center_list = center_list
        sum_rows_center = sum([x[1] for x in center_list if isinstance(x, tuple)])
        avg_rows_centers = sum_rows_center/len([x for x in center_list if isinstance(x, tuple)])
        for each in missing_type:
            min_diff = np.abs(type_list-each)
            min_v, min_ind = np.min(min_diff), np.argmin(min_diff)+1
            if min_ind > 2:  # 引导点在分割点右侧
                if each > min_ind:  # 需要补全的字符在引导点右侧
                    new_center_list[each-1] = (center_list[min_ind-1][0]+min_v*self.__dist_between_chars[1], avg_rows_centers)
                else:  # 左侧
                    if each <= 2:
                        new_center_list[each-1] = (center_list[min_ind-1][0]-self.__dist_between_chars[0]-(min_v-1)*self.__dist_between_chars[1], avg_rows_centers)
                    else:
                        new_center_list[each-1] = (center_list[min_ind-1][0]-min_v*self.__dist_between_chars[1], avg_rows_centers)
            else:  # 在分割点左侧
                if each > min_ind:  # 需要补全的字符在引导点右侧
                    if each > 2:
                        new_center_list[each-1] = (center_list[min_ind-1][0]+self.__dist_between_chars[0]+(min_v-1)*self.__dist_between_chars[1], avg_rows_centers)
                    else:
                        new_center_list[each-1] = (center_list[min_ind-1][0]+min_v*self.__dist_between_chars[1], avg_rows_centers)
                else:  # 左侧
                    new_center_list[each-1] = (center_list[min_ind-1][0]-min_v*self.__dist_between_chars[1], avg_rows_centers)
        return new_center_list


    def __getCharsMsgFinely(self, type_list, bin_img, center_set, width_set, height_set):
        '''

        :param type_list:
        :param bin_img:
        :param center_set:
        :param width_set:
        :param height_set:
        :return:
        '''
        missing_types = [ind for ind, types in enumerate(type_list) if types == -10]
        current_types = [ind for ind, types in enumerate(type_list) if types != -10]
        new_center_set = np.zeros((len(type_list), 2), np.int16)
        center_set = np.array(center_set).reshape(len(center_set), 2)
        for each in missing_types:
            posx, posy = center_set[each, 0], center_set[each, 1]
            width, height = width_set[each], height_set[each]
            min_row, max_row = int(posy-height/2), int(posy+height/2)
            min_col, max_col = int(posx-width/2), int(posx+width/2)
            roi_img = bin_img[min_row:max_row, min_col:max_col]
            m = cv2.moments(roi_img)
            xc, yc = m['m10']/m['m00'], m['m01']/m['m00']
            new_center_set[each, :] = (xc+min_col, yc+min_row)
        for each in current_types:
            new_center_set[each, :] = (int(center_set[each, 0]), int(center_set[each, 1]))
        return new_center_set


    # @test.timeit
    def __getCharsBoxingMsg(self, real_contours):
        '''
        :: 从真实字符轮廓中得到字符box的信息
        :param real_contours: 真实字符轮廓
        :return: 字符box信息，包括中心点list， 理想长度， 理想宽度，如果区域没有字符信息，则抛出异常。
        '''
        centers_loc = []
        width_set = []
        height_set = []
        for each in real_contours:
            max_row, min_row, max_col, min_col = max(each[:, :, 1]), min(each[:, :, 1]), max(each[:, :, 0]), min(each[:, :, 0])
            width_set.append(max_col-min_col)
            height_set.append(max_row-min_row)
            loc = ((max_col+min_col)/2, (max_row+min_row)/2)
            centers_loc.append(loc)
        width_ideal = np.max(width_set)
        height_ideal = np.max(height_set)
        return centers_loc, width_ideal, height_ideal


    def __deleteSmallCharRegions(self, img):
        '''
        :: 除去图片中明显不是字符的区域
        :param img: 车牌二值图
        :return: 真实的字符轮廓
        '''
        _, chars_contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        real_contours = []
        for each in chars_contours:
            dcol, drow = max(each[:, :, 0])-min(each[:, :, 0]), max(each[:, :, 1])-min(each[:, :, 1])
            # dcol+0.0001 防止出现除0异常
            if self.__chars_wh_lower <= (drow/(dcol+0.0001)) <= self.__chars_wh_upper:
                area = cv2.contourArea(each)
                if self.__chars_area_lower <= area <= self.__chars_area_upper:
                    real_contours.append(each)
        return real_contours

    # @test.timeit
    # TODO(FesianXu) 优化中心点的位置和box的窗口信息，考虑用自适应聚集收敛的方法。
    def plateSegment(self, img, isGray=False, gray_img=None):
        '''
        :: 对二值车牌进行字符分割
        :param img: 车牌的二值图
        :return: roi_set 返回车牌分割的字符图像
        '''
        chars_contours = self.__deleteSmallCharRegions(img)
        # centers_loc, width_ideal, height_ideal = self.__getCharsBoxingMsg(chars_contours)
        try:
            centers_loc, width_ideal, height_ideal = self.__getCharsBoxingMsg(chars_contours)
        except ValueError:
            return None
        type_list = self.__decideCharsTypes(centers_loc)
        type_list, center_list = self.__mergeDuplicateContours(type_list, centers_loc)
        center_list = self.__getMissingCharsMsgRoughly(type_list, center_list)
        width_set, height_set = [width_ideal]*7, [height_ideal]*7  # 简单粗暴处理
        center_list = self.__getCharsMsgFinely(type_list, img, center_list, width_set, height_set)
        if isGray and gray_img is not None:
            roi_set = self.__cutTheChars(gray_img, center_list, width_set, height_set, isGray)
        else:
            roi_set = self.__cutTheChars(img, center_list, width_set, height_set, isGray)

        ######################### DEBUG ########################################
        # loc = np.array(center_list).reshape(len(center_list), 2)
        # fig1 = plt.figure()
        # ax1 = fig1.add_subplot(111, aspect='equal')
        # plt.imshow(img, cmap ='gray')
        # for each in chars_contours:
        #     plt.scatter(each[:,:,0], each[:,:,1], color='r')
        # plt.scatter(loc[:,0], loc[:,1], color='b')
        # posx, posy = loc[:, 0], loc[:, 1]
        # m_loc = []
        # for ind, each in enumerate(posx):
        #     ax1.add_patch(
        #         patches.Rectangle(
        #             (posx[ind]-width_set[ind]/2, posy[ind]-height_set[ind]/2),   # (x,y)
        #             width_set[ind],          # width
        #             height_set[ind],          # height
        #             fill=False,
        #             linewidth=2.0,
        #             edgecolor=(0.8, 0.8, 0.1)
        #         )
        #     )
        #     min_row, max_row = int(posy[ind]-height_set[ind]/2), int(posy[ind]+height_set[ind]/2)
        #     min_col, max_col = int(posx[ind]-width_set[ind]/2), int(posx[ind]+width_set[ind]/2)
        #     roi_img = img[min_row:max_row, min_col:max_col]
        #     m = cv2.moments(roi_img)
        #     xc, yc = m['m10']/m['m00'], m['m01']/m['m00']
        #     m_loc.append((xc+min_col, yc+min_row))
        # m_loc = np.array(m_loc).reshape(len(m_loc), 2)
        # plt.scatter(m_loc[:,0], m_loc[:,1], color='g')
        # posx, posy = m_loc[:, 0], m_loc[:, 1]
        # for ind, each in enumerate(posx):
        #     ax1.add_patch(
        #         patches.Rectangle(
        #             (posx[ind]-width_set[ind]/2, posy[ind]-height_set[ind]/2),   # (x,y)
        #             width_set[ind],          # width
        #             height_set[ind],          # height
        #             fill=False,
        #             linewidth=2.0,
        #             edgecolor=(0.20, 0.91, 0.13),
        #             linestyle='--'
        #         )
        #     )
        # plt.show()
        ######################### DEBUG ########################################

        return roi_set


########################################################################################################################

import PlateRecognize.PlateDetector as PlateDetector
path = 'F:/opencvjpg/'
name = '1156.jpg'
file_name = path+name
img = cv2.imread(file_name)
is_saveGray = False
det = PlateDetector.PlateDetector()
seg = PlateSegment(det.getImageNormalizedWidth(), det.getImageNormalizedHeight())

# @test.timeit
def main():
    img_mat = det.getPlateRegion(img)
    img_out_bin, img_out_gray = det.plateCorrect(img_mat)
    for ind, each in enumerate(img_out_bin):
        roi_set = seg.plateSegment(each, is_saveGray)
        # res = CharsPredict.predict_chars(roi_set)

        # for ind_i, each_i in enumerate(roi_set):
        #     plt.subplot(1, 7, ind_i+1)
        #     plt.imshow(each_i, cmap ='gray')
        # plt.show()


if __name__ == '__main__':
    cv2.setUseOptimized(True)
    main()

