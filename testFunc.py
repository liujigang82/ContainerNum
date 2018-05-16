import sys
import cv2
import numpy as np
import pytesseract
from preprocessing.preprocessing import get_perspective_transformed_im
import glob
#sys.path.append('C:\\Users\\RT\\Documents\\git\\ContainerNum\\utils')
sys.path.append('F:\\Projects\\ConainerNum\\ContainerNum\\utils')
import textRec, drawRect, kmeans, get_contours, calculateAngle

#pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract'
pytesseract.pytesseract.tesseract_cmd = 'Tesseract-OCR/tesseract'
mser = cv2.MSER_create()
mser.setMaxArea(750)

# global para
threshold_width = 1/4
threshold_height = 1/3

def intersection(box1,box2):
  x = max(box1[0], box2[0])
  y = max(box1[1], box2[1])
  w = min(box1[0]+box1[2], box2[0]+box2[2]) - x
  h = min(box1[1]+box1[3], box2[1]+box2[3]) - y
  if w<0 or h<0: return (0,0,0,0)
  return (x, y, w, h)

def not_inside(bbox, coords):
    if len(coords) == 0:
        return True
    else:
        for coord in coords:
            box = cv2.boundingRect(coord)
            # compare box and bbox
            intersects = intersection(box, bbox)
            if intersects != (0,0,0,0):

            #if box[0]<= bbox[0] and box[1] <= bbox[1] \
            #    and box[0]+box[2]>=bbox[0]+bbox[2] and box[1]+box[3] >= bbox[1]+bbox[3]:
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
    #numbers = sum(c.isdigit() for c in str)
    #words = sum(c.isalpha() for c in str)
    #spaces = sum(c.isspace() for c in str)
    return abs(4-words) + abs(7-numbers)


def isAlpha(str):
    for character in str:
        if not character.isalpha():
            return False
    return True

def find_index_word(str):
    length = 0
    word = ""
    sub_str_list = str.split(" ")
    for sub in sub_str_list:
        if isAlpha(sub) and len(sub) > length:
            length = len(sub)
            word = sub
    if length > 0:
        index = str.index(word)
        str = str[index:len(str)]
    return str, len(word)-1


def result_refine(str):
    for char in str:
        if not char.isdigit() and not char.isalpha() and not char.isspace():
            str = str.replace(char, "")
    print("str:", str)
    '''
    try:
        index = str.lower().index("u")
    except:
        index = 0
    if index == 0:
        try:
            index = str.lower().index("v")
        except:
            index = 0
    '''
    str, index = find_index_word(str)
    '''
    index = 0
    for i in range(len(str)):
        if not str[i].isalpha():
            index = i
            break
    '''
    text = str[0:index+1]
    digits = str[index+1:len(str)]

    print("text:", text, "digits:", digits)
    text = [character for character in text if character.isalpha()]
    text = "".join(item for item in text)
    digits = [digit for digit in digits if digit.isdigit()]
    digits = "".join(item for item in digits)
    print(index, text, digits)
    str = text + " " + digits
    '''
    if index-3 >= 0:
        str = str[index-3: len(str)]
    '''

    return str


def is_text(str):
    words = sum(c.isalpha() for c in str)
    if words == len(str):
        return True
    else:
        return False


def myfind(y, x):
    return [ a for a in range(len(y)) if y[a] == x]

def containDigAlph(str):
    for i in range(len(str)):
        if str[i].isdigit() or str[i].isalpha():
            return True
    return False

def get_image_patch(canvas, tesseract_data, result):
    t_boundary = 10
    for i in range(len(tesseract_data["text"])):
        itemList = tesseract_data["text"]
        text = [character for character in itemList[i] if character.isalpha()]
        text = "".join(item for item in text)
        if text in result and len(text) >= 2 and is_text(text):
            print(text)
            level_index = myfind(tesseract_data["level"], tesseract_data["level"][i])
            page_index = myfind(tesseract_data["level"], tesseract_data["level"][i])
            block_index = myfind(tesseract_data["block_num"], tesseract_data["block_num"][i])
            par_index = myfind(tesseract_data["par_num"], tesseract_data["par_num"][i])
            line_index = myfind(tesseract_data["line_num"], tesseract_data["line_num"][i])
            word_index =  tesseract_data["word_num"][i]

            index = [i for i in level_index if
                     i in page_index and i in block_index and i in par_index and i in line_index]
            print(index)
            to_remove = []
            for i in range(len(index)):
                if not containDigAlph(tesseract_data["text"][index[i]]):
                    to_remove.append(index[i])
                    break
            for i in range(len(to_remove)):
                index.remove(to_remove[i])


            #index = index[word_index-1:len(index)]
            print("after:", index)
            left1 = tesseract_data["left"][index[0]]
            left2 = tesseract_data["left"][index[len(index) - 1]]
            top1 = tesseract_data["top"][index[0]]
            top2 = tesseract_data["top"][index[len(index) - 1]]

            width1 = tesseract_data["width"][index[0]]
            width2 = tesseract_data["width"][index[len(index) - 1]]
            height1 = tesseract_data["height"][index[0]]
            height2 = tesseract_data["height"][index[len(index) - 1]]

            left = min(left1, left2)
            top = min(top1, top2)
            right = max(left1 + width1, left2 + width2)
            bottom = max(top1 + height1, top2 + height2)

            top_tmp = top - t_boundary if top - t_boundary > 0 else 0

            canvas = canvas[top_tmp:bottom + t_boundary, left - t_boundary:right + t_boundary]

    return canvas

def preprocessing_im(img):
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
    # color to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # perspective transform
    gray = get_perspective_transformed_im(gray)
    #cv2.imshow("perspective", gray)
    cv2.imshow("after deskew", gray)
    textRec.text_detection_MSER(gray)
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
        if x > width * threshold_width and y < height * threshold_height  and not_inside(bbox, coords) and w  <width/15 and h < height/15:
            coords.append(coord)

    canvas3 = np.zeros_like(gray)
    for cnt in coords:
        xx = cnt[:, 0]
        yy = cnt[:, 1]
        color = 255
        canvas3[yy, xx] = color

    #kernel = np.ones((2, 2), np.uint8)
    #canvas3 = cv2.erode(canvas3, kernel, iterations=1)
    canvas3 = cv2.GaussianBlur(canvas3,(3,3),0)
    cv2.imshow("canvas", canvas3)
    #canvas3 = calculateAngle.calculateAngle(canvas3)
    image_str = pytesseract.image_to_string(canvas3)
    print("result:",image_str)

    min_conf = 100
    result = ""
    for line in image_str.splitlines():
        cur_conf = str_confidence(line)
        if cur_conf < min_conf:
            result = line
            min_conf = cur_conf
    result = result_refine(result)
    print("results:", result)

    tesseract_data = pytesseract.image_to_data(canvas3, output_type="dict")
    print( pytesseract.image_to_data(canvas3))
    canvas3 = get_image_patch(canvas3, tesseract_data, result)
    canvas3 = calculateAngle.calculateAngle(canvas3)
    cv2.imshow("image patch", canvas3)
    print("refined results:", pytesseract.image_to_string(canvas3))


    return result
'''

for file in glob.glob("img/*.jpg"):

    img = cv2.imread(file)
    #cv2.imshow("image", img)
    gray = preprocessing_im(img)
    #result = postprocessing(gray)
    print(file, ":", postprocessing(gray))
    cv2.waitKey(0)
'''
img = cv2.imread("img/0088.jpg")
cv2.imshow("image", img)
gray = preprocessing_im(img)
postprocessing(gray)
cv2.waitKey(0)


