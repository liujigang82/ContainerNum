import numpy as np
import cv2
from matplotlib import pyplot as plt

def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


img = cv2.imread("../img3/105.jpg")
print(img.shape)
cv2.imshow("Red", img[:, :, 2])
cv2.imshow("Green", img[:, :, 1])
cv2.imshow("Blue", img[:, :, 0])
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(gray)
cv2.imshow("gray", gray)
cv2.imshow("CLAHE", cl1)


hist = cv2.calcHist([img],[0],None,[256],[0,256])
plt.plot(hist,color = "r")
plt.xlim([0, 256])
cv2.imshow("img", img)
ad_img = adjust_gamma(img, 0.5)
cv2.imshow("adjust", ad_img)
hist = cv2.calcHist([ad_img],[0],None,[256],[0,256])
plt.plot(hist,color = "b")
plt.xlim([0, 256])
plt.show()
