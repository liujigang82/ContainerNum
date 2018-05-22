import sys
import cv2
import numpy as np
import pytesseract
from preprocessing.preprocessing import get_perspective_transformed_im, detect_shape
from utils import get_image_patch, calculateAngle, find_index_word
import glob
#sys.path.append('F:\\Projects\\ConainerNum\\ContainerNum\\utils')
#import textRec, drawRect, kmeans, get_contours

#pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract'
pytesseract.pytesseract.tesseract_cmd = 'Tesseract-OCR/tesseract'
mser = cv2.MSER_create()
mser.setMaxArea(750)

# global para
threshold_width = 1/5
threshold_height = 1/3

def intersection(a,b):
  x = max(a[0], b[0])
  y = max(a[1], b[1])
  w = min(a[0]+a[2], b[0]+b[2]) - x
  h = min(a[1]+a[3], b[1]+b[3]) - y
  if w<0 or h<0: return (0,0,0,0)
  return (x, y, w, h)


def not_inside(bbox, coords):
    if len(coords) == 0:
        return True
    else:
        for coord in coords:
            box = cv2.boundingRect(coord)
            # compare box and bbox
            #if box[0]<= bbox[0] and box[1] <= bbox[1] \
            #    and box[0]+box[2]>=bbox[0]+bbox[2] and box[1]+box[3] >= bbox[1]+bbox[3]:
            intersects = intersection(box, bbox)
            if intersects != (0, 0, 0, 0):
                return False
        return True


def contour_rec_ara(contour):
    bbox = cv2.boundingRect(contour)
    x, y, w, h = bbox
    return w*h


def str_confidence(str):
    str_list = str.split(" ")
    numbers = 0
    words = 0
    for item in str_list:
        if len(item) == 4 and item[len(item)-1].lower() == "u":
            return 0
        numbers = sum(c.isdigit() for c in item) if sum(c.isdigit() for c in item)>numbers else numbers
        words = sum(c.isalpha() for c in item) if sum(c.isalpha() for c in item) > words else words

    return abs(4-words) +abs(7-numbers)


def result_refine(image_str):
    min_conf = 100
    str = ""
    for line in image_str.splitlines():
        cur_conf = str_confidence(line)
        if cur_conf < min_conf:
            str = line
            min_conf = cur_conf

    for char in str:
        if not char.isdigit() and not char.isalpha() and not char.isspace():
            str = str.replace(char, "")

    str, index = find_index_word(str)
    index = 0
    for i in range(len(str)):
        if not str[i].isalpha():
            index = i
            break
    text = str[0:index+1]
    digits = str[index+1:len(str)]

    text = [character for character in text if character.isalpha()]
    text = "".join(item for item in text)
    digits = [digit for digit in digits if digit.isdigit()]
    digits = "".join(item for item in digits)
    return text + " " + digits


def is_text(str):
    words = sum(c.isalpha() for c in str)
    if words == len(str):
        return True
    else:
        return False


def preprocessing_im(img):
    # resize
    height, width, depth = img.shape
    imgScale = 600/width
    newX,newY = img.shape[1]*imgScale, img.shape[0]*imgScale
    img = cv2.resize(img, (int(newX),int(newY)))
    # histogram equalization
    #img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    #img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    #img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    img = cv2.fastNlMeansDenoisingColored(img, None, 7, 7, 7, 21)
    # color to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # perspective transform
    gray = get_perspective_transformed_im(gray)
    #cv2.imshow("perspective", gray)
    #cv2.imshow("after deskew", gray)
    #textRec.text_detection_MSER(gray)
    #binary
    #edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    #cv2.imshow("edge",edges)
    #(threshold, im_bw) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return gray


def postprocessing(gray):
    coords = []
    coordinates, bboxes = mser.detectRegions(gray)
    coordinates = sorted(coordinates, key=contour_rec_ara, reverse=True)
    (height, width) = gray.shape[:2]
    for coord in coordinates:
        bbox = cv2.boundingRect(coord)
        x, y, w, h = bbox
        if x > width * threshold_width and y < height * threshold_height and w  <width/15 and h < height/15 and not_inside(bbox, coords):
            coords.append(coord)

    canvas3 = np.zeros_like(gray)
    for cnt in coords:
        xx = cnt[:, 0]
        yy = cnt[:, 1]
        color = 255
        canvas3[yy, xx] = color

    canvas3 = cv2.GaussianBlur(canvas3, (3, 3), 0)
    #cv2.imshow("canvas", canvas3)
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
    cv2.imshow("image", img)
    gray = preprocessing_im(img)
    print("after preprocessing...")
    return postprocessing(gray)

def find_rect(file):
    img = cv2.imdecode(np.fromfile(file, dtype=np.uint8), -1)
    gray = preprocessing_im(img)

    smoothed_img = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.addWeighted(gray, 1.5, smoothed_img, -0.5, 0)


    mser = cv2.MSER_create()
    mser.setMinArea(50)
    mser.setMaxArea(750)
    contours, bboxes = mser.detectRegions(gray)

    coords = []
    contours = sorted(contours, key=contour_rec_ara, reverse=True)

    (height, width) = gray.shape[:2]
    backtorgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    for c in contours:
        bbox = cv2.boundingRect(c)
        x, y, w, h = bbox
        if x > width * threshold_width and y < height * threshold_height and float(w / h) <= 1 and w < width / 15 and h < height / 15 and not_inside(bbox, coords):
            coords.append(c)

    canvas = np.zeros_like(gray)
    for cnt in coords:
        xx = cnt[:, 0]
        yy = cnt[:, 1]
        color = 255
        canvas[yy, xx] = color

    cv2.imshow("canvas", canvas)

    im2, contours, hierarchy = cv2.findContours(canvas.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        c = cv2.convexHull(c)
        shape = detect_shape(c)
        if shape == "rectangle" or shape == "square":
            cv2.drawContours(backtorgb, [c], -1, (0, 0,255), 2)
        if shape == "triangle":
            cv2.drawContours(backtorgb, [c], -1, (255, 0, 0), 2)
        if shape == "pentagon":
            cv2.drawContours(backtorgb, [c], -1, (255, 255, 0), 2)
        if shape == "circle":
            cv2.drawContours(backtorgb, [c], -1, (255, 0, 255), 2)
        if shape == "unidentified":
            print("unidentified")
            cv2.drawContours(backtorgb, [c], -1, (255, 255, 0), 2)



    cv2.imshow("Image", backtorgb)