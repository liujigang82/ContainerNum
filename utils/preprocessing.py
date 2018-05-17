import cv2
import numpy as np

def resize_im(image):
    height, width, depth = image.shape
    imgScale = 600/width
    newX,newY = image.shape[1]*imgScale, image.shape[0]*imgScale
    image = cv2.resize(image, (int(newX),int(newY)))
    return image

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged