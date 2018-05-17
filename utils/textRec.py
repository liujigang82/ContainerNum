#!/usr/bin/python3
# 2017.10.05 10:52:58 CST
# 2017.10.05 13:27:18 CST
"""
Text detection with MSER, and fill with random colors for each detection.
"""

import numpy as np
import cv2


def text_detection_MSER(img):
    ## Read image and change the color space
    im_shape = img.shape
    if len(im_shape)==2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ## Get mser, and set parameters
    mser = cv2.MSER_create()
    #mser.setMinArea(100)
    #mser.setMaxArea(2000)

    ## Do mser detection, get the coodinates and bboxes
    coordinates, bboxes = mser.detectRegions(gray)
    
    ## colors
    colors = [[43, 43, 200], [43, 75, 200], [43, 106, 200], [43, 137, 200], [43, 169, 200], [43, 200, 195], [43, 200, 163], [43, 200, 132], [43, 200, 101], [43, 200, 69], [54, 200, 43], [85, 200, 43], [116, 200, 43], [148, 200, 43], [179, 200, 43], [200, 184, 43], [200, 153, 43], [200, 122, 43], [200, 90, 43], [200, 59, 43], [200, 43, 64], [200, 43, 95], [200, 43, 127], [200, 43, 158], [200, 43, 190], [174, 43, 200], [142, 43, 200], [111, 43, 200], [80, 43, 200], [43, 43, 200]]

    ## Fill with random colors
    np.random.seed(0)
    canvas1 = img.copy()
    canvas3 = np.zeros_like(img)

    for cnt in coordinates:
        xx = cnt[:,0]
        yy = cnt[:,1]
        color = colors[np.random.choice(len(colors))]
        canvas1[yy, xx] = color
        canvas3[yy, xx] = color
    #cv2.imshow('result',canvas3)
    cv2.imshow("canvas1", canvas1)


'''
## Filter the coordinates
vis = img.copy()
coords = []
for coord in coordinates:
    bbox = cv2.boundingRect(coord)
    x,y,w,h = bbox
    #if w< 10 or h < 10 or w/h > 5 or h/w > 5:
     #   continue
    coords.append(coord)
'''