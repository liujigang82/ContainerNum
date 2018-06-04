import cv2
from preprocessing_local import resize_im
import textRec
img = cv2.imread('../img/img3/9.jpg')
gray_1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = resize_im(img)
print(gray_1.shape)

'''
img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
img = cv2.fastNlMeansDenoisingColored(img, None, 7, 7, 7, 21)
'''
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("before", gray)
smoothed_img = cv2.GaussianBlur(gray, (3, 3), 0)
#cv2.imshow("smooth",smoothed_img)
gray_result = cv2.addWeighted(gray, 1.5 , smoothed_img, -0.5, 0)

textRec.text_detection_MSER(gray_result)

cv2.imshow("after", gray_result)
cv2.waitKey(0)
