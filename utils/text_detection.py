#!/usr/bin/python

import sys
import os

import cv2 as cv
import numpy as np


img = cv.imread("../img/0002.jpg")
# for visualization
vis = img.copy()

textSpotter = cv.text.TextDetectorCNN_create("textbox.prototxt", "TextBoxes_icdar13.caffemodel")
rects, outProbs = textSpotter.detect(img);
vis2 = img.copy()
thres = 0.1

for r in range(np.shape(rects)[0]):
    if outProbs[r] > thres:
        rect = rects[r]
        cv.rectangle(vis2, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 0, 0), 2)

cv.imshow("Text detection result2", vis2)

'''
gray = cv.cvtColor(vis, cv.COLOR_BGR2GRAY)
erc1 = cv.text.loadClassifierNM1('./trained_classifierNM1.xml')
er1 = cv.text.createERFilterNM1(erc1)

erc2 = cv.text.loadClassifierNM2('./trained_classifierNM2.xml')
er2 = cv.text.createERFilterNM2(erc2)

regions = cv.text.detectRegions(gray,er1,er2)

#Visualization
rects = [cv.boundingRect(p.reshape(-1, 1, 2)) for p in regions]

print(rects)
for rect in rects:
  cv.rectangle(vis, rect[0:2], (rect[0]+rect[2],rect[1]+rect[3]), (0, 0, 0), 2)
for rect in rects:
  cv.rectangle(vis, rect[0:2], (rect[0]+rect[2],rect[1]+rect[3]), (255, 255, 255), 1)
cv.imshow("Text detection result22", img)

'''





# Extract channels to be processed individually
channels = cv.text.computeNMChannels(img)
# Append negative channels to detect ER- (bright regions over dark background)
cn = len(channels) - 1
for c in range(0, cn):
    channels.append((255 - channels[c]))

# Apply the default cascade classifier to each independent channel (could be done in parallel)
print("Extracting Class Specific Extremal Regions from " + str(len(channels)) + " channels ...")
print("    (...) this may take a while (...)")
for channel in channels:

    erc1 = cv.text.loadClassifierNM1('./trained_classifierNM1.xml')
    er1 = cv.text.createERFilterNM1(erc1, 16, 0.00015, 0.13, 0.2, True, 0.1)

    erc2 = cv.text.loadClassifierNM2('./trained_classifierNM2.xml')
    er2 = cv.text.createERFilterNM2(erc2, 0.5)

    regions = cv.text.detectRegions(channel, er1, er2)

    rects = cv.text.erGrouping(img, channel, [r.tolist() for r in regions])
    # rects = cv.text.erGrouping(img,channel,[x.tolist() for x in regions], cv.text.ERGROUPING_ORIENTATION_ANY,'../../GSoC2014/opencv_contrib/modules/text/samples/trained_classifier_erGrouping.xml',0.5)

    # Visualization
    for r in range(0, np.shape(rects)[0]):
        rect = rects[r]
        #cv.rectangle(vis, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 0), 2)
        cv.rectangle(vis, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 255), 1)

# Visualization
cv.imshow("Text detection result", vis)


cv.waitKey(0)
