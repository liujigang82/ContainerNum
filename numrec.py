import sys
import cv2
import numpy as np
import pytesseract
from preprocessing.preprocessing import get_perspective_transformed_im,resize_im
from postprocessing import get_binary_text_ROI, get_image_patch, calculateAngle
from textProcessing import result_refine
from textProcessing import find_index_word
import glob
#sys.path.append('F:\\Projects\\ConainerNum\\ContainerNum\\utils')
#import textRec, drawRect, kmeans, get_contours

#pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract'
pytesseract.pytesseract.tesseract_cmd = 'Tesseract-OCR/tesseract'\

def preprocessing_im(img):
    # resize
    img = resize_im(img)
    # histogram equalization
    #img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    #img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    #img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    #img = cv2.fastNlMeansDenoisingColored(img, None, 7, 7, 7, 21)
    # color to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # perspective transform
    gray = get_perspective_transformed_im(gray)
    smoothed_img = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.addWeighted(gray, 1.5, smoothed_img, -0.5, 0)

    #cv2.imshow("perspective", gray)
    #cv2.imshow("after deskew", gray)
    #textRec.text_detection_MSER(gray)
    #binary
    #edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    #cv2.imshow("edge",edges)
    #(threshold, im_bw) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return gray


def postprocessing(gray):

    canvas3 = get_binary_text_ROI(gray)
    cv2.imshow("canvas", canvas3)
    image_str = pytesseract.image_to_string(canvas3)
    #tesseract_data = pytesseract.image_to_data(canvas3, output_type="dict")

    result = result_refine(image_str)
    print("results:", result)

    tesseract_data = pytesseract.image_to_data(canvas3, output_type="dict")
    canvas3 = get_image_patch(canvas3, tesseract_data, result)
    #cv2.imshow("imagepatch", canvas3)
    canvas3 = calculateAngle(canvas3)
    refined_result = pytesseract.image_to_string(canvas3)
    print(refined_result)
    refined_result = result_refine(refined_result)
    print("refined results:", refined_result)
    '''
    for i in range(len(tesseract_data["text"])):
        itemList = tesseract_data["text"]
        if itemList[i] in result and len(itemList[i]) >= 2 and is_text(
                itemList[i]):
            h_roi = int(tesseract_data["height"][i] + 10)
            w_roi = int(h_roi * 6.5)
            left = tesseract_data["left"][i] - 10
            top = tesseract_data["top"][i] - 10
            img_patch = gray[top:top + h_roi + 10, left:left + w_roi + 10]
            #cv2.imshow("patch", img_patch)
            #drawRect.drawRect(img_patch)
            #kmeans.kmeans(img_patch)
            #kmeans.kmeans(img_patch)
            #print("refined results:", pytesseract.image_to_string(kmeans.kmeans(img_patch)))
            print("refined results:", pytesseract.image_to_string(img_patch))
    '''
    return  refined_result
'''

for file in glob.glob("img/*.jpg"):

    img = cv2.imread(file)
    #cv2.imshow("image", img)
    gray = preprocessing_im(img)
    #result = postprocessing(gray)
    print(file, ":", postprocessing(gray))
    cv2.waitKey(0)
'''
'''
img = cv2.imread("img/IMG_4693.jpg")
# cv2.imshow("image", img)
gray = preprocessing_im(img)
print("after preprocessing...")
postprocessing(gray)
'''
def num_rec(file):
    img = cv2.imdecode(np.fromfile(file, dtype=np.uint8), -1)
    #cv2.imshow("image", img)
    gray = preprocessing_im(img)
    print("preprocessing done...")
    return postprocessing(gray)
