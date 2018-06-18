import cv2
import numpy as np
import pytesseract
import glob
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract'


def auto_canny(image, sigma=0.33): #0.33
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    cv2.imshow("edges", edged)
    # return the edged image
    return edged

def resize_im(image):
    height, width, depth = image.shape
    imgScale = 800/width
    newX,newY = image.shape[1]*imgScale, image.shape[0]*imgScale
    image = cv2.resize(image, (int(newX),int(newY)))
    return image

def preprocessing(img):
    # resize
    # histogram equalization
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    #img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    img = cv2.fastNlMeansDenoisingColored(img, None, 7, 7, 7, 21)
    #cv2.imshow("after denoise", img)
    # color to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray", gray)
    # perspective transform
    #gray = deskew(gray.copy(), compute_skew(gray))
    #cv2.imshow("after deskew", gray)

    #binary
    #edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    edges = auto_canny(gray)
    cv2.imshow("edge",edges)
    #(threshold, im_bw) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return edges

def compute_intersection(line_first, line_sec):
    x = (np.sin(line_first[1])*line_sec[0]-np.sin(line_sec[1])*line_first[0])/(np.sin(line_first[1])*np.cos(line_sec[1])-np.sin(line_sec[1])*np.cos(line_first[1]))
    if np.sin(line_sec[1]) !=0:
        y = (line_sec[0]-x*np.cos(line_sec[1]))/np.sin(line_sec[1])
    elif  np.sin(line_first[1]) !=0:
        y = (line_first[0] - x * np.cos(line_first[1])) / np.sin(line_first[1])
    else:
        return None
    return [x, y]


def dist(point1, point2):
    return np.sqrt((point1[0]-point2[0])*(point1[0]-point2[0])+(point1[1]-point2[1])*(point1[1]-point2[1]))

def compute_perspective_matrix(vertical_params, horizontal_params):
    np_vert = np.abs(np.array(vertical_params))
    np_hori = np.abs(np.array(horizontal_params))

    #
    index_rho, index_theta = np_vert.argmax(axis=0)
    vert_max = vertical_params[index_rho]

    index_rho, index_theta = np_hori.argmax(axis=0)
    hori_max = horizontal_params[index_rho]

    index_rho, index_theta = np_vert.argmin(axis=0)
    vert_min = vertical_params[index_rho]

    index_rho, index_theta = np_hori.argmin(axis=0)
    hori_min = horizontal_params[index_rho]
    print(vert_max, vert_min, hori_max, hori_min)

    p_left_top = compute_intersection(hori_min, vert_min)
    p_right_top = compute_intersection(hori_min, vert_max)
    p_left_bottom = compute_intersection(hori_max, vert_min)
    p_right_bottom = compute_intersection(hori_max, vert_max)
    print(p_left_top, p_right_top, p_left_bottom, p_right_bottom)
    pts1 = np.float32([p_left_top, p_right_top,p_left_bottom, p_right_bottom])
    pts2 = np.float32([p_left_top, [p_left_top[0] + dist(p_left_top, p_right_top), p_left_top[1]],
                [p_left_top[0], p_left_top[1] + dist(p_left_top, p_left_bottom)],
                [p_left_top[0] + dist(p_left_top, p_right_top), p_left_top[1] + dist(p_left_top, p_left_bottom)]])

    pers_matrix = cv2.getPerspectiveTransform(pts1, pts2)
    return pers_matrix

#Your image path i-e receipt path

img = cv2.imread('../img/img3/39.jpg')
img  = resize_im(img)

gray_vis = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = preprocessing(img)

# detect regions in gray scale image
height, width = gray.shape
# lines = cv2.HoughLinesP(gray, rho = 1, theta=np.pi/180, threshold= 100, minLineLength=height /2.0, maxLineGap=30)
lines = cv2.HoughLines(gray, rho=1, theta=np.pi / 180, threshold=160)
vertical_params = []
horizontal_params = []
#print("lines:", lines)
print("lines", lines)
for line in lines:

    #print("line:", line)
    for rho, theta in line:
        a = np.cos(theta)
        b = np.sin(theta)
        # print("x value when y = 200:", (rho-b*200)/a)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        if theta * 180 / np.pi < 20 or theta * 180 / np.pi > 160:
            vertical_params.append([rho, theta])
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if theta * 180 / np.pi > 70 and theta * 180 / np.pi < 110:
            # print("horizontal~~~~")
            horizontal_params.append([rho, theta])
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
#vertical_params = np.array(vertical_params)
print("vertical:", vertical_params)
print("horizontal:", horizontal_params)
cv2.imshow("line image", img)
if len(vertical_params)>1 and len(horizontal_params) > 1:
    h, w = gray_vis.shape
    cv2.imshow("before perspective:", gray_vis)
    dst = cv2.warpPerspective(gray_vis, compute_perspective_matrix(vertical_params, horizontal_params), (w, h))
    cv2.imshow("perspective", dst)

cv2.waitKey(33)


'''
i = 0
for file in glob.glob("../img/*.jpg"):
    i = i+1
    print("~~~~~~~~")
    print(file)
    vertical_params = []
    horizontal_params = []
    img = cv2.imread(file)
    img = resize_im(img)
    gray_vis = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = preprocessing(img)
    #detect regions in gray scale image
    height, width = gray.shape
    #lines = cv2.HoughLinesP(gray, rho = 1, theta=np.pi/180, threshold= 100, minLineLength=height /2.0, maxLineGap=30)
    lines = cv2.HoughLines(gray, rho=1, theta =np.pi/180, threshold = 200)
    x_val = []
    for line in lines:
        for rho, theta in line:
            if theta*180/np.pi < 30 or theta*180/np.pi > 150:
                    vertical_params.append([rho, theta])
                    a = np.cos(theta)
                    b = np.sin(theta)
                    #print("x value when y = 200:", (rho-b*200)/a)
                    x_val.append((rho-b*200)/a)
                    x0 = a*rho
                    y0 = b*rho
                    x1 = int(x0 + 1000*(-b))
                    y1 = int(y0 + 1000*(a))
                    x2 = int(x0 - 1000*(-b))
                    y2 = int(y0 - 1000*(a))
                    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
            if theta*180/np.pi > 60 and theta*180/np.pi < 125:
                #print("horizontal~~~~")
                a = np.cos(theta)
                b = np.sin(theta)
                horizontal_params.append([rho, theta])
                #print("x value when y = 200:", (rho - b * 200) / a)
                #x_val.append((rho - b * 200) / a)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    if len(vertical_params) > 1 and len(horizontal_params) > 1:
        h, w = gray_vis.shape
        dst = cv2.warpPerspective(gray_vis, compute_perspective_matrix(vertical_params, horizontal_params), (w, h))
        #cv2.imshow("perspective", dst)
        cv2.imwrite( "deskewed_%04d.jpg"%i, dst)
    else:
        cv2.imwrite("deskewed_%04d.jpg"%i, img)

    cv2.waitKey(33)

'''

















'''
#detect regions in gray scale image
height, width = gray.shape
#lines = cv2.HoughLinesP(gray, rho = 1, theta=np.pi/180, threshold= 100, minLineLength=height /2.0, maxLineGap=30)
lines = cv2.HoughLines(gray, rho=1, theta =np.pi/180, threshold = 200)
x_val = []
for line in lines:
    for rho, theta in line:
        if theta*180/np.pi < 30 or theta*180/np.pi > 150:
                a = np.cos(theta)
                b = np.sin(theta)
                print("x value when y = 200:", (rho-b*200)/a)
                x_val.append((rho-b*200)/a)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))
                cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
        else:
            print("horizontal~~~~")
            a = np.cos(theta)
            b = np.sin(theta)
            #print("x value when y = 200:", (rho - b * 200) / a)
            #x_val.append((rho - b * 200) / a)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

x_val = sorted(x_val)
print("x value:", x_val)
dist = []
for i in range(len(x_val)-1):
    dist.append(x_val[i+1]-x_val[i])
    print("dist:", x_val[i+1]-x_val[i])

cv2.imshow("line image", img)
(im2, cnts, _) = cv2.findContours(gray.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
coordinates = sorted(cnts, key=cv2.contourArea, reverse=True)

for contour in coordinates[:20]:
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = h/w
    area = cv2.contourArea(contour)
    cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)
#cv2.imshow("drawcontours", img)

(height, width) = gray.shape[:2]

coords = []
for coord in coordinates[:20]:
    bbox = cv2.boundingRect(coord)
    x, y, w, h = bbox
    coords.append(coord)
'''

'''
canvas3 = np.zeros_like(gray)
print("coords",coords)
for cnt in coords:
    xx = cnt[:,0]
    yy = cnt[:,1]
    color = 255
    canvas3[yy, xx] = color

cv2.imshow('canvas', canvas3)

    for x1, y1, x2, y2 in line:
        if y2 != y1:
            cur_angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            if cur_angle > 45 or cur_angle < -45:
                print("~~~~~~", x1, y1, x2, y2)
                cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
'''
cv2.waitKey(0)
