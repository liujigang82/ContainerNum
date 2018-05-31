import cv2
import numpy as np
import pytesseract
from preprocessing.preprocessing import get_perspective_transformed_im,resize_im
from postprocessing import get_binary_text_ROI, get_image_patch, calculateAngle
from textProcessing import result_refine, final_refine, str_confidence

pytesseract.pytesseract.tesseract_cmd = 'Tesseract-OCR/tesseract'\

def preprocessing_im(img):
    # resize
    print(img.shape)
    img = resize_im(img)
    # histogram equalization
    #img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    #img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    #img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    #img = cv2.fastNlMeansDenoisingColored(img, None, 7, 7, 7, 21)
    # color to gray
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
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
    print(image_str)

    min_conf = 100
    result = ""
    for line in image_str.splitlines():
        cur_conf = str_confidence(line)
        if cur_conf < min_conf:
            result = line
            min_conf = cur_conf
    result = result_refine(result)
    '''
    tesseract_data = pytesseract.image_to_data(canvas3, output_type="dict")
    canvas3 = get_image_patch(canvas3, tesseract_data, result)
    #cv2.imshow("imagepatch", canvas3)
    canvas3 = calculateAngle(canvas3)
    refined_result = pytesseract.image_to_string(canvas3)
    refined_result = result_refine(refined_result)
    '''
    refined_result = final_refine(result)

    return  refined_result

def num_rec(file):
    img = cv2.imdecode(np.fromfile(file, dtype=np.uint8), -1)
    cv2.imshow("img",img)
    gray = preprocessing_im(img)
    print("preprocessing done...")
    return postprocessing(gray)
