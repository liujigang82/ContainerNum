import pytesseract

import cv2
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract'

def preprocessing(img):
    # resize
    height, width, depth = img.shape
    imgScale = 600/width
    newX,newY = img.shape[1]*imgScale, img.shape[0]*imgScale
    img = cv2.resize(img, (int(newX),int(newY)))
    # histogram equalization
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    #img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    img = cv2.fastNlMeansDenoisingColored(img, None, 7, 7, 7, 21)
    #cv2.imshow("after denoise", img)
    # color to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # perspective transform
    #gray = deskew(gray.copy(), compute_skew(gray))
    #cv2.imshow("after deskew", gray)

    #binary
    #edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    #cv2.imshow("edge",edges)
    #(threshold, im_bw) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return gray

img = cv2.imread('../img/keanms.jpg')
gray = preprocessing(img)
print("~~~~start~~~~~~~~~~~~`")
print(pytesseract.image_to_string(gray))
print(pytesseract.image_to_data(gray))
print(pytesseract.image_to_boxes(gray))

print("~~~~end~~~~~~~~~~~~`")
