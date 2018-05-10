import numpy as np
import cv2

def drawRect(img):
    im_shape = img.shape
    if len(im_shape)==2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ## Get mser, and set parameters
    mser = cv2.MSER_create()
    mser.setMinArea(50)
    mser.setMaxArea(500)
    coordinates, bboxes = mser.detectRegions(gray)
    print("coordinates:", len(coordinates))
    for p in coordinates:
        xmax, ymax = np.amax(p, axis=0)
        xmin, ymin = np.amin(p, axis=0)
        cv2.rectangle(img, (xmin, ymax), (xmax, ymin), (0, 255, 255), 1)

    canvas3 = np.zeros_like(gray)
    for cnt in coordinates:
        xx = cnt[:, 0]
        yy = cnt[:, 1]
        color = 255
        canvas3[yy, xx] = color
    cv2.imshow("rect", img)
    cv2.imshow("dd",canvas3)