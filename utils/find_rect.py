import cv2
import numpy as np
from preprocessing import auto_canny


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
        #if shape == "square" or shape == "rectangle":
         #   print("~~~~~~~~~~:",  peri)
            #print("points:", c)


    # if the shape is a pentagon, it will have 5 vertices
    elif len(approx) == 5:
        shape = "pentagon"

    # otherwise, we assume the shape is a circle
    else:
        shape = "circle"

    # return the name of the shape
    return shape

#82,
imageName = "../img/0014.jpg"

img = cv2.imdecode(np.fromfile(imageName,dtype = np.uint8),-1)
#img = cv2.imread(imageName.encode('gbk'),-1)
img = resize_im(img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = auto_canny(gray)

cv2.imshow("image", gray)
cv2.imshow("binary", edges)

'''
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
edges = cv2.dilate(edges, kernel)

cv2.imshow("dilate", edges)
'''
im2, contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

'''
mser = cv2.MSER_create()
mser.setMinArea(50)
mser.setMaxArea(750)
contours, bboxes = mser.detectRegions(edges)
'''

print(contours[0])
print(cv2.convexHull(contours[0]))

# loop over the contours
for c in contours:
    # compute the center of the contour, then detect the name of the
    # shape using only the contour
    #cv2.drawContours(img, [c], -1, (0, 0, 255), 2)

    # multiply the contour (x, y)-coordinates by the resize ratio,
    # then draw the contours and the name of the shape on the image

    hulls = cv2.convexHull(c)
    #hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in c]
    #print(hulls)

    shape = detect(c)
    if  shape == "square":# or shape == "rectangle":
        cv2.drawContours(img, [c], -1, (0, 255, 0), 2)

    # show the output image
    cv2.imshow("Image", img)

cv2.waitKeyEx(0)
