import cv2

def image_enhancement(image):
    equ = cv2.equalizeHist(image)
    cv2.imshow("gray", image)
    cv2.imshow("equ", equ)


img = cv2.imread('../img/0031.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
image_enhancement(gray)

bitwise = cv2.bitwise_not(gray)
cv2.imshow('bitwise',bitwise)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(gray)
cv2.imshow('clahe_2.jpg',cl1)

cv2.waitKeyEx(0)