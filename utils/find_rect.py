import cv2
import numpy as np
from preprocessing import auto_canny, not_inside, contour_rec_ara, get_perspective_transformed_im

# global para
threshold_width = 1/4
threshold_height = 1/3

def resize_im(image):
    height, width, depth = image.shape
    imgScale = 600 / width
    newX, newY = image.shape[1] * imgScale, image.shape[0] * imgScale
    image = cv2.resize(image, (int(newX), int(newY)))
    return image


def detect(c):
    # initialize the shape name and approximate the contour
    shape = "unidentified"

    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)

    if cv2.isContourConvex(approx) and abs(cv2.contourArea(c))<80:
        return shape
    # if the shape is a triangle, it will have 3 vertices
    if len(approx) == 3:
        shape = "triangle"

    # if the shape has 4 vertices, it is either a square or
    # a rectangle
    elif len(approx) == 4:
        # compute the bounding box of the contour and use the
        # bounding box to compute the aspect ratio
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)

        # a square will have an aspect ratio that is approximately
        # equal to one, otherwise, the shape is a rectangle
        #shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
        shape = "square" if ar >= 0.50 and ar <= 0.8 else "rectangle"

    # if the shape is a pentagon, it will have 5 vertices
    elif len(approx) == 5:
        shape = "pentagon"

    # otherwise, we assume the shape is a circle
    else:
        shape = "circle"

    # return the name of the shape
    return shape

#82,
imageName = "../img/0074.jpg"

img = cv2.imdecode(np.fromfile(imageName,dtype = np.uint8),-1)

img = resize_im(img)


cv2.imshow("image1", img)

'''
img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
'''
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray = get_perspective_transformed_im(gray)

smoothed_img = cv2.GaussianBlur(gray, (3, 3), 0)
gray = cv2.addWeighted(gray, 1.5, smoothed_img, -0.5, 0)


#im2, contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)


mser = cv2.MSER_create()
#mser.setMinArea(50)
mser.setMaxArea(750)
contours, bboxes = mser.detectRegions(gray)


(height, width) = gray.shape[:2]
# loop over the contours
coords = []
contours = sorted(contours, key=contour_rec_ara, reverse=True)

for c in contours:
    bbox = cv2.boundingRect(c)
    x, y, w, h = bbox
    shape = detect(c)
    if x > width * threshold_width and y < height * threshold_height and float(w / h) <= 1 and float(w / h) >= 0.25 and w  < width/15 and h < height/15 and not_inside(bbox, coords) :
        coords.append(c)
        #cv2.drawContours(img, [c], -1, (0, 255,0), 1)


canvas = np.zeros_like(gray)
for cnt in coords:
    xx = cnt[:, 0]
    yy = cnt[:, 1]
    color = 255
    canvas[yy, xx] = color

cv2.imshow("canvas", canvas)
backtorgb = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
im2, contours, hierarchy = cv2.findContours(canvas.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
for c in contours:
    shape = detect(c)
    if  shape == "square" or shape == "rectangle":
       cv2.drawContours(backtorgb, [c], -1, (0, 0, 255), 2)

cv2.imshow("Image", backtorgb)
cv2.waitKeyEx(0)
