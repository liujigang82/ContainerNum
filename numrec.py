import cv2
import numpy as np
import pytesseract
import glob

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract'
mser = cv2.MSER_create()
mser.setMinArea(50)
mser.setMaxArea(500)
flag_all = True


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged


def compute_skew(image):
    image = cv2.bitwise_not(image)
    height, width = image.shape
    edges = auto_canny(image)
    cv2.imshow("edge", edges)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=height /2.0, maxLineGap=20)
    angle = 0.0
    abs_angle = 90.0
    count = 0

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                if y2 != y1 and x2!=x1:
                    cur_angle = np.arctan2(y2 - y1, x2 - x1)
                    if cur_angle > np.pi/4 or cur_angle < -np.pi/4:
                        count = count + 1
                        angle += cur_angle*180/np.pi
                        abs_angle += abs(cur_angle * 180 / np.pi)
    else:
        return 0

    if angle > 0:
        return (abs_angle / count - 90)
    else:
        return( 90 - abs_angle/count)

def deskew(image, angle):
    rows, cols = image.shape
    center = (rows/2, cols/2)
    root_mat = cv2.getRotationMatrix2D(center, angle, 1)
    rotated = cv2.warpAffine(image, root_mat, (cols, rows), flags=cv2.INTER_CUBIC)
    return cv2.getRectSubPix(rotated, (cols, rows), center)



def str_confidence(str):
    numbers = sum(c.isdigit() for c in str)
    words = sum(c.isalpha() for c in str)
    spaces = sum(c.isspace() for c in str)
    others = len(str) - numbers - words - spaces

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
    gray = deskew(gray.copy(), compute_skew(gray))
    cv2.imshow("after deskew", gray)

    #binary
    #edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    edges = auto_canny(gray)
    #cv2.imshow("edge",edges)
    #(threshold, im_bw) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return gray

def postprocessing(gray):
    coords = []
    coordinates, bboxes = mser.detectRegions(gray)
    (height, width) = gray.shape[:2]
    for coord in coordinates:
        bbox = cv2.boundingRect(coord)
        x, y, w, h = bbox
        if x > width / 4 and y < height / 3:
            coords.append(coord)
    canvas3 = np.zeros_like(gray)
    for cnt in coords:
        xx = cnt[:, 0]
        yy = cnt[:, 1]
        color = 255
        canvas3[yy, xx] = color
    image_str = pytesseract.image_to_string(canvas3)
    tesseract_data = pytesseract.image_to_data(canvas3,output_type= "dict")
    min_conf = 100
    result = ""
    for line in image_str.splitlines():
        cur_conf = str_confidence(line)
        if cur_conf < min_conf:
            result = line
            min_conf = cur_conf
    result = result_refine(result)
    print("results:", result)
    for i in range(len(tesseract_data["text"])):
        itemList = tesseract_data["text"]
        if itemList[i] in result and len(itemList[i]) >= 2 and is_text(
                itemList[i]):
            h_roi = int(tesseract_data["height"][i] + 10)
            w_roi = int(h_roi * 6.5)
            left = tesseract_data["left"][i] - 10
            top = tesseract_data["top"][i] - 10
            img_patch = gray[top:top + h_roi + 10, left:left + w_roi + 10]
            print("--refined results:", pytesseract.image_to_string(img_patch))
    return result

'''
for file in glob.glob("img/*.jpg"):
    print("~~~~~~~~")
    print(file)
    img = cv2.imread(file)
    cv2.imshow("image", img)
    gray = preprocessing(img)
    #result = postprocessing(gray)
    #print(file, ":", postprocessing(gray))
'''



img = cv2.imread("img/IMG_4693.jpg")
cv2.imshow("image", img)
gray = preprocessing(img)
cv2.waitKey(0)
vis = gray.copy()
'''
#detect regions in gray scale image
coords = []
coordinates, bboxes = mser.detectRegions(gray)

(height, width) = gray.shape[:2]

for coord in coordinates:
    #print("~~~~~", coord, "~~~~")
    bbox = cv2.boundingRect(coord)
    x, y, w, h = bbox
    if x > width/4 and y < height/3:
        coords.append(coord)


#### ----- to get correct region ----------------
canvas3 = np.zeros_like(gray)
for cnt in coords:
    xx = cnt[:,0]
    yy = cnt[:,1]
    color = 255
    canvas3[yy, xx] = color

cv2.imshow('canvas', canvas3)


#cv2.imshow("text only", text_only)

image_str  = pytesseract.image_to_string(canvas3)
print(image_str)

min_conf = 100
result = ""
for line in image_str.splitlines():
    cur_conf = str_confidence(line)
    if cur_conf < min_conf:
        result = line
        min_conf = cur_conf

result = result_refine(result)
print("results:", result)

tesseract_data = pytesseract.image_to_data(canvas3, output_type= "dict")
#print(tesseract_data)
#print(tesseract_data["text"])
img_patch = []
for i in range(len(tesseract_data["text"])):
    if tesseract_data["text"][i] in result and len(tesseract_data["text"][i]) >= 2 and is_text(tesseract_data["text"][i]):
        h_roi = int(tesseract_data["height"][i] + 10)
        w_roi = int(h_roi * 6.5)

        left = tesseract_data["left"][i]-10
        top = tesseract_data["top"][i]-10
        img_patch = gray[top:top+h_roi+10, left:left+w_roi+10]

        cv2.rectangle(vis, (left, top),
                      (left + w_roi + 10, top + h_roi+10), (0, 255, 255), 1)
        #th = cv2.adaptiveThreshold(img_patch, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
        cv2.imshow("patch", img_patch)
        print("refined results:", pytesseract.image_to_string(img_patch))

cv2.imshow("refined", vis)
cv2.waitKey(0)
'''

#(im2, cnts, _) = cv2.findContours(gray.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#coordinates = sorted(cnts, key=cv2.contourArea, reverse=True)

'''
coords = np.column_stack(np.where(threshold > 0))
angle = cv2.minAreaRect(coords)[-1]

# the `cv2.minAreaRect` function returns values in the
# range [-90, 0); as the rectangle rotates clockwise the
# returned angle trends to 0 -- in this special case we
# need to add 90 degrees to the angle
if angle < -45:
	angle = -(90 + angle)

else:
	angle = -angle

(h, w) = img.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated = cv2.warpAffine(img, M, (w, h),
	flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

cv2.imshow("rotated", rotated)
'''

'''
edges = cv2.Canny(gray,50,150,apertureSize = 3)
cv2.imshow('edge', edges)
lines = cv2.HoughLines(edges,1,np.pi/180,200)
print(lines.size)
for rho,theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
'''

'''
print(coordinates)
for contour in coordinates[:2000]:
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = h/w
    area = cv2.contourArea(contour)
    cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)
cv2.imshow("drawcontours", img)
'''

'''
contours = []


for p in coords:
    xmax, ymax = np.amax(p, axis=0)
    xmin, ymin = np.amin(p, axis=0)
    if xmin > width/4 and ymin < height/3:
        object = []
        object.append([xmin, ymin])
        object.append([xmin, ymax])
        object.append([xmax, ymax])
        object.append([xmax, ymin])
        contours.append(object)
        cv2.rectangle(vis, (xmin,ymax), (xmax,ymin), (0, 255, 255), 1)

#hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

#cv2.imshow('img', vis)

mask = np.zeros((vis.shape[0], vis.shape[1], 1), dtype=np.uint8)

for contour in np.array(contours):
    cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)

#this is used to find only text regions, remaining are ignored
text_only = cv2.bitwise_and(gray, gray, mask=mask)
'''