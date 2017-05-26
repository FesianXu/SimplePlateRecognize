# !/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'FesianXu'
__date__ = '2017/5/20'
__version__ = ''


import cv2
import numpy as np
import time
import functools

def timeit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        begin = time.clock()
        res = func(*args, **kwargs)
        end = time.clock()
        print('time cost = ', end-begin, 'in function = ', func.__name__)
        return res
    return wrapper



########################################################################################################################

windowName = 'img'
coordinate = []
def mouse_cb(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        coordinate.append([x, y])

@timeit
def correct(after):
     img_correct = cv2.warpPerspective(img, homograghy, (180, 48))
     return img_correct

if __name__ == '__main__':
    path = 'F:/opencvjpg/41.jpg'
    img = cv2.imread(path)
    cv2.namedWindow(windowName)
    cv2.setMouseCallback(windowName, mouse_cb, [img]) # mouse call back
    while True:
        cv2.imshow(windowName, img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    after = [[0, 0], [180, 0], [0, 48], [180, 48]]
    after = np.array(after, np.float32)
    coordinate = np.array(coordinate, np.float32)
    homograghy = cv2.getPerspectiveTransform(coordinate, after)
    print(homograghy)
    img_correct = correct(after)
    cv2.imshow('new', img_correct)
    cv2.waitKey(-1)
    cv2.destroyAllWindows()