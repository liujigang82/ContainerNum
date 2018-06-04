import cv2
import numpy as np

img = cv2.imread('../img/0004.jpg', cv2.IMREAD_GRAYSCALE)

def dist(point1, point2):
    return np.sqrt((point1[0]-point2[0])*(point1[0]-point2[0])+(point1[1]-point2[1])*(point1[1]-point2[1]))

def compute_skew(image):
    image = cv2.bitwise_not(image)
    height, width = image.shape
    vis = image.copy()
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    #cv2.imshow("edge", edges)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=height /4.0, maxLineGap=20)
    angle = 0.0
    abs_angle = 0.0
    count = 0

    max_angle = 0.0
    min_angle = 90.0
    max_point1 = []
    max_point2 = []
    min_point1 = []
    min_point2 = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            if y2 != y1 and x2!=x1:
                cur_angle = np.arctan2(y2 - y1, x2 - x1)*180/np.pi
                if cur_angle > 45 or cur_angle < -45:
                    cv2.line(vis, (x1, y1), (x2, y2), 255, 2)
                    count = count + 1
                    angle += cur_angle*180/np.pi
                    abs_angle += abs(cur_angle * 180 / np.pi)
                    if cur_angle > 0:
                        cur_angle = cur_angle-90
                    else:
                        cur_angle = 90-abs(cur_angle)

                    if max_angle < cur_angle:
                        max_angle = cur_angle
                        if y1 < y2:
                            max_point1 = [x1, y1]
                            max_point2 = [x2, y2]
                        else:
                            max_point2 = [x1, y1]
                            max_point1 = [x2, y2]
                    if min_angle > cur_angle:
                        min_angle = cur_angle
                        if y1 < y2:
                            min_point1 = [x1, y1]
                            min_point2 = [x2, y2]
                        else:
                            min_point2 = [x1, y1]
                            min_point1 = [x2, y2]
    cv2.imshow("line detection", vis)
    print("angle max and min", max_angle, min_angle, max_point1, max_point2, min_point1, min_point2)
    pts1 = np.float32([max_point1, max_point2, min_point1, min_point2])
    pts2 = np.float32([max_point1, [max_point1[0], max_point1[1]+dist(max_point1,max_point2)], min_point1, [min_point1[0], min_point1[1]+dist(min_point1,min_point2)]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    cv2.line(image, (max_point1[0],max_point1[1]), (max_point2[0],max_point2[1]), 255, 2)
    cv2.line(image, (min_point1[0],min_point1[1]), (min_point2[0],min_point2[1]), 255, 2)
    dst = cv2.warpPerspective(image, M, image.shape)
    cv2.imshow("perspective", dst)
    '''

    for line in lines:
        for x1, y1, x2, y2 in line:
            if y2 != y1 and x2!=x1:
                cur_angle = np.arctan2(y2 - y1, x2 - x1)
                if cur_angle > np.pi/4 or cur_angle < -np.pi/4:
                    print(cur_angle*180/np.pi)
                    print("length", np.sqrt((y1-y2)*(y1-y2)+(x1-x2)*(x1-x2)))
                    cv2.line(image, (x1, y1), (x2, y2), 255, 2)
                    count = count + 1
                    angle += cur_angle*180/np.pi
                    abs_angle += abs(cur_angle * 180 / np.pi)

    cv2.imshow("line detection", image)
   '''
    print("angle...", angle / count, count)
    if angle > 0:
        return (abs_angle / count - 90)
    else:
        return( 90 - abs_angle/count)



def deskew(image, angle):
    #image = cv2.bitwise_not(image)
    non_zero_pixels = cv2.findNonZero(image)
    center, wh, theta = cv2.minAreaRect(non_zero_pixels)
    print("center", center, angle)
    root_mat = cv2.getRotationMatrix2D(center, angle, 1)
    rows, cols = image.shape
    rotated = cv2.warpAffine(image, root_mat, (cols, rows), flags=cv2.INTER_CUBIC)
    #cv2.imshow("rotated", rotated)
    return cv2.getRectSubPix(rotated, (cols, rows), center)

angle = compute_skew(img)
print(angle)
deskewed_image = deskew(img.copy(), angle)


cv2.imshow('original', img)
cv2.imshow('deskew', deskewed_image)
cv2.waitKey(0)
