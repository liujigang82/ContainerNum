import cv2
import numpy as np

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged

def compute_skew(image):
    image = cv2.bitwise_not(image)
    height, width = image.shape
    edges = auto_canny(image)
    cv2.imshow("edge", edges)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=height /2.0, maxLineGap=20)
    angle = 0.0
    abs_angle = 90.0
    count = 0

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                if y2 != y1 and x2!=x1:
                    cur_angle = np.arctan2(y2 - y1, x2 - x1)
                    if cur_angle > np.pi/4 or cur_angle < -np.pi/4:
                        count = count + 1
                        angle += cur_angle*180/np.pi
                        abs_angle += abs(cur_angle * 180 / np.pi)
    else:
        return 0

    if angle > 0:
        return (abs_angle / count - 90)
    else:
        return( 90 - abs_angle/count)

def deskew(image, angle):
    rows, cols = image.shape
    center = (rows/2, cols/2)
    root_mat = cv2.getRotationMatrix2D(center, angle, 1)
    rotated = cv2.warpAffine(image, root_mat, (cols, rows), flags=cv2.INTER_CUBIC)
    return cv2.getRectSubPix(rotated, (cols, rows), center)
