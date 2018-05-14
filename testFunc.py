import sys
import cv2
import numpy as np
import pytesseract
from preprocessing.preprocessing import get_perspective_transformed_im
import glob
sys.path.append('C:\\Users\\RT\\Documents\\git\\ContainerNum\\utils')
#sys.path.append('F:\\Projects\\ConainerNum\\ContainerNum\\utils')
import textRec, drawRect, kmeans, get_contours, calculateAngle

#pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract'
pytesseract.pytesseract.tesseract_cmd = 'Tesseract-OCR/tesseract'
mser = cv2.MSER_create()
mser.setMaxArea(700)

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
    numbers = sum(c.isdigit() for c in str)
    words = sum(c.isalpha() for c in str)
    spaces = sum(c.isspace() for c in str)
    return abs(4-words) +abs(7-numbers)


def result_refine(str):
    for char in str:
        if not char.isdigit() and not char.isalpha() and not char.isspace():
            str = str.replace(char, "")
    return str


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
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    #img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    #img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    img = cv2.fastNlMeansDenoisingColored(img, None, 7, 7, 7, 21)
    # color to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # perspective transform
    gray = get_perspective_transformed_im(gray)
    #cv2.imshow("perspective", gray)
    cv2.imshow("after deskew", gray)
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
        if x > width * threshold_width and y < height * threshold_height and w  <width/20 and h < height/20 and not_inside(bbox, coords):
            coords.append(coord)

    canvas3 = np.zeros_like(gray)
    for cnt in coords:
        xx = cnt[:, 0]
        yy = cnt[:, 1]
        color = 255
        canvas3[yy, xx] = color

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
    #print(tesseract_data)
    for i in range(len(tesseract_data["text"])):
        itemList = tesseract_data["text"]
        if itemList[i] in result and len(itemList[i]) >= 2 and is_text(
                itemList[i]):
            h_roi = int(tesseract_data["height"][i] + 15)
            w_roi = int(h_roi * 5.5)
            left = tesseract_data["left"][i] - 15
            top = tesseract_data["top"][i] - 15
            img_patch = canvas3[top:top + h_roi + 15, left:left + w_roi + 15]
            #img_patch = cv2.equalizeHist(img_patch)
            cv2.imshow("patch", img_patch)
            cv2.imwrite("tmp.jpg", img_patch)
            #tmp = drawRect.drawRect(img_patch)
            #kmeans.kmeans(img_patch)
            #kmeans.kmeans(img_patch)
            #print("refined results:", pytesseract.image_to_string(kmeans.kmeans(img_patch)))
            print("refined results:", pytesseract.image_to_string(img_patch))

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

img = cv2.imread("img/0007.jpg")
cv2.imshow("image", img)
gray = preprocessing_im(img)
postprocessing(gray)
cv2.waitKey(0)


