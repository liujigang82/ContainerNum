import cv2
import numpy as np
import pytesseract

def dist(point1, point2):
    return np.sqrt((point1[0]-point2[0])*(point1[0]-point2[0])+(point1[1]-point2[1])*(point1[1]-point2[1]))


image = cv2.imread("../img/img3/76.jpg")
cv2.imshow("image", image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
max_point1 = [122, 258]
max_point2 = [302, 263]
min_point1 = [82, 535]
min_point2 = [309, 535]

pts1 = np.float32([max_point1, max_point2, min_point1, min_point2])
pts2 = np.float32([max_point1, [max_point1[0]+dist(max_point1, max_point2), max_point1[1]], [max_point1[0], max_point1[1]+dist(max_point1, min_point1)],
                   [max_point1[0]+dist(max_point1, max_point2), max_point1[1]+dist(max_point1, min_point1)]])

print(pts1)
print(pts2)
M = cv2.getPerspectiveTransform(pts1, pts2)
print(M)
h, w = gray.shape

dst = cv2.warpPerspective(gray, M, (w, h))

cv2.imshow("perspective", dst)
cv2.waitKey(0)