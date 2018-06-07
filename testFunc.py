import sys
import cv2
import pytesseract
import numpy as np
from preprocessing import get_perspective_transformed_im, resize_im
from postprocessing import get_binary_text_ROI
#sys.path.append('C:\\Users\\RT\\Documents\\git\\ContainerNum\\utils')
sys.path.append('F:\\Projects\\ConainerNum\\ContainerNum')
from utils_test import textRec, drawRect, get_contours, calculateAngle
from textProcessing import str_confidence, result_refine, final_refine
#pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract'
pytesseract.pytesseract.tesseract_cmd = 'Tesseract-OCR/tesseract'

# global para
threshold_width = 1/4
threshold_height = 1/3


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
    textRec.text_detection_MSER(gray)
    # perspective transform
    gray, flag = get_perspective_transformed_im(gray)
    if not flag:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        gray, flag = get_perspective_transformed_im(gray)

    smoothed_img = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.addWeighted(gray, 1.5, smoothed_img, -0.5, 0)


    #cv2.imshow("perspective", gray)
    cv2.imshow("after deskew", gray)

    #binary
    #edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    #cv2.imshow("edge",edges)
    #(threshold, im_bw) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return gray


def postprocessing(gray):
    canvas3 = get_binary_text_ROI(gray)
    #canvas3 = calculateAngle.calculateAngle(canvas3)

    image_str = pytesseract.image_to_string(canvas3)
    print("result:", image_str)
    cv2.imshow("canvas :", canvas3)

    min_conf = 100
    result = ""
    for line in image_str.splitlines():
        cur_conf = str_confidence(line)
        if cur_conf < min_conf:
            result = line
            min_conf = cur_conf
    result = result_refine(result)
    print("results:", result)
    result, flag = final_refine(result)
    print("refined results:", result)

    '''
    tesseract_data = pytesseract.image_to_data(canvas3, output_type="dict")
    print( pytesseract.image_to_data(canvas3))
    canvas3 = get_image_patch(canvas3, tesseract_data, result)
    canvas3 = calculateAngle.calculateAngle(canvas3)
    cv2.imshow("image patch", canvas3)
    result = pytesseract.image_to_string(canvas3)
    print("1", result)
    result = result_refine(result)
    print("2", result)
    '''
    return result


file = "img/img1/0017.jpg" #oolu.jpg
#img = cv2.imread("img2/CMAU.jpg")  #0022
img = cv2.imdecode(np.fromfile(file, dtype=np.uint8), -1)
#cv2.imshow("image", img)
gray = preprocessing_im(img)
#get_contour_list(gray)
postprocessing(gray)
cv2.waitKey(0)


'''

for file in glob.glob("img/*.jpg"):
    img = cv2.imread(file)
    #cv2.imshow("image", img)
    gray = preprocessing_im(img)
    #result = postprocessing(gray)
    print(file, ":", postprocessing(gray))
    cv2.waitKey(0)
'''

