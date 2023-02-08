# https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

print('Loading up...')

for fg_factor in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:

    img = cv.imread('LiveDead Image Analysis/MFGTMP_210413160001_A01f02d0.PNG')
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

    # cv.imshow('image', img)
    # cv.imwrite('image.png', img)

    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)
    # sure background area
    sure_bg = cv.dilate(opening,kernel,iterations=3)
    # Finding sure foreground area
    dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
    ret, sure_fg = cv.threshold(dist_transform,fg_factor*dist_transform.max(),255,0)

    cv.imwrite(f'sure_fg/{fg_factor}.png', sure_fg)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg,sure_fg)

    # Marker labelling
    ret, markers = cv.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0

    markers = cv.watershed(img,markers)
    img[markers == -1] = [255,0,0]

    # print(ret)
    # cv.imwrite(f'ret/{fg_factor}.png', ret)
    cv.imwrite(f'markers/{fg_factor}.png', markers)

    # print(markers)
    # print(len(markers))
    print(f'{fg_factor}: {np.max(markers)}')

    cv.imwrite(f'markers{fg_factor}.png', img)