import cv2
import numpy as np

def is_solid_box(cnt):
    x, y, w, h = cv2.boundingRect(cnt)
    cnt_hull = cv2.convexHull(cnt)
    print("~~~",cv2.contourArea(cnt),  cv2.contourArea(cnt_hull), w*h )
    if float(h)/float(w) > 3:
        return False
    if float(cnt.shape[0]) / float(w*h) < 0.7: #cv2.contourArea(cnt)
        return  False
    '''
    area = cv2.contourArea(cnt_hull)
    if area == 0:
        return True
    if float(cnt.shape[0]) / float(area) < 0.9:
        return False
    '''
    return True


imageName = "../img/test/solid_box.jpg" #solid_box

img = cv2.imdecode(np.fromfile(imageName,dtype = np.uint8),-1)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


mser = cv2.MSER_create()
mser.setMaxArea(750)
contours, bboxes = mser.detectRegions(gray)
#im2, contours, hierarchy = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
n_white_pix = np.sum(gray == 255)
print(n_white_pix)

backtorgb = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
backtorgb2 = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)

for c in contours:
    bbox = cv2.boundingRect(c)
    x, y, w, h = bbox
    cv2.rectangle(backtorgb, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cnt_hull = cv2.convexHull(c)
    cv2.drawContours(backtorgb2, [cnt_hull], -1, (0, 0, 255), 2)

    print("~~~", cv2.contourArea(c), cv2.contourArea(cnt_hull), w * h, c.shape[0])
    print(w, h)

cv2.imshow("rect", backtorgb)
cv2.imshow("rect2", backtorgb2)
cv2.waitKey(0)