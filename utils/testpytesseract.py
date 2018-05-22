import pytesseract
import numpy as np
import cv2
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract'

t_boundary = 5

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


def is_text(str):
    words = sum(c.isalpha() for c in str)
    if words == len(str):
        return True
    else:
        return False

def myfind(y, x):
    return [ a for a in range(len(y)) if y[a] == x]

img = cv2.imread('../img/patch_0009.png')
cv2.imshow("image", img)
#gray = preprocessing(img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, gray = cv2.threshold(gray,200,255,cv2.THRESH_BINARY)
cv2.imshow("gray", gray)
#kernel = np.ones((2,2),np.uint8)
#gray = cv2.erode(gray,kernel,iterations = 1)
#cv2.imshow("erosion", gray)

print("~~~~start~~~~~~~~~~~~`")

result = pytesseract.image_to_string(gray)
print(result)

print("~~~~~~~~~~~~~~~~~")
print(pytesseract.image_to_data(gray))
tesseract_data = pytesseract.image_to_data(gray, output_type="dict")
# print(tesseract_data)
for i in range(len(tesseract_data["text"])):
    itemList = tesseract_data["text"]
    if itemList[i] in result and len(itemList[i]) >= 2 and is_text(itemList[i]):
        level_index = myfind(tesseract_data["level"], tesseract_data["level"][i])
        page_index = myfind(tesseract_data["level"], tesseract_data["level"][i])
        block_index = myfind(tesseract_data["block_num"],tesseract_data["block_num"][i])
        par_index = myfind(tesseract_data["par_num"],tesseract_data["par_num"][i])
        line_index = myfind(tesseract_data["line_num"], tesseract_data["line_num"][i])

        index = [i for i in level_index if i in page_index and i in block_index and i in par_index and i in line_index]
        print("index", index)

        left1 = tesseract_data["left"][index[0]]
        left2 = tesseract_data["left"][index[len(index)-1]]

        top1 = tesseract_data["top"][index[0]]
        top2 = tesseract_data["top"][index[len(index)-1]]

        width1 = tesseract_data["width"][index[0]]
        width2 = tesseract_data["width"][index[len(index)-1]]

        height1 = tesseract_data["height"][index[0]]
        height2 = tesseract_data["height"][index[len(index)-1]]

        left = min(left1, left2)
        top = min(top1, top2)

        right = max(left1+width1, left2+width2)
        bottom = max(top1+height1, top2+height2)



        #h_roi = int(tesseract_data["height"][i] + 15)
        #w_roi = int(h_roi * 5.5)
        #left = tesseract_data["left"][i] - 15
        #top = tesseract_data["top"][i] - 15
        #img_patch = gray[top:top + h_roi + 15, left:left + w_roi + 15]
        img_patch = gray[top-t_boundary:bottom+t_boundary, left-t_boundary:right+t_boundary]
        cv2.imshow("patch", img_patch)
        print("refined results:", pytesseract.image_to_string(img_patch))


print("~~~~end~~~~~~~~~~~~`")
cv2.waitKey(0)
