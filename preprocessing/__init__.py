import numpy as np
import cv2
from postprocessing import get_text_line

angle_threshold_line_detection = 20

def dist(point1, point2):
    return np.sqrt((point1[0]-point2[0])*(point1[0]-point2[0])+(point1[1]-point2[1])*(point1[1]-point2[1]))


def compute_intersection(line_first, line_sec):
    x = (np.sin(line_first[1])*line_sec[0]-np.sin(line_sec[1])*line_first[0])/(np.sin(line_first[1])*np.cos(line_sec[1])-np.sin(line_sec[1])*np.cos(line_first[1]))
    if np.sin(line_sec[1]) !=0:
        y = (line_sec[0]-x*np.cos(line_sec[1]))/np.sin(line_sec[1])
    elif  np.sin(line_first[1]) !=0:
        y = (line_first[0] - x * np.cos(line_first[1])) / np.sin(line_first[1])
    else:
        return None
    return [x, y]


def hist_equalization(img):
    # histogram equalization
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    img = cv2.fastNlMeansDenoisingColored(img, None, 7, 7, 7, 21)
    return img


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged


def get_horizontal_vertical_lines(gray):
    vertical_params=[]
    gray = auto_canny(gray)
    #cv2.imshow("edge", gray)
    #detect regions in gray scale image
    height, width = gray.shape
    lines = cv2.HoughLines(gray, rho=2, theta =np.pi/180, threshold = int(height/3)) #threshold = int(height/3)

    if lines is not None:
        for line in lines:
            for rho, theta in line:
                if theta*180/np.pi < angle_threshold_line_detection or theta*180/np.pi > 180 - angle_threshold_line_detection:
                    vertical_params.append([rho, theta])
                #if theta*180/np.pi > 90-angle_threshold_line_detection and theta*180/np.pi < 90 + angle_threshold_line_detection:
                #    horizontal_params.append([rho, theta])
    return vertical_params

def draw_line(gray, rho , theta):
    a = np.cos(theta)
    b = np.sin(theta)
    # print("x value when y = 200:", (rho-b*200)/a)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv2.line(gray, (x1, y1), (x2, y2), (255, 0, 0), 2)
    if theta * 180 / np.pi < 20 or theta * 180 / np.pi > 160:
        cv2.line(gray, (x1, y1), (x2, y2), (0, 255, 0), 2)
    if theta * 180 / np.pi > 70 and theta * 180 / np.pi < 110:
        cv2.line(gray, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return gray

def check_line_selected(gray, vertical_params, horizontal_params):


    np_vert = np.abs(np.array(vertical_params))
    np_hori = np.abs(np.array(horizontal_params))
    pers_matrix =  []
    backtorgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    #for rho, theta in vertical_params:
    #    backtorgb = draw_line(backtorgb, rho, theta)

    index_rho, index_theta = np_vert.argmax(axis=0)
    vert_max = vertical_params[index_rho]
    backtorgb = draw_line(backtorgb, vert_max[0], vert_max[1])

    index_rho, index_theta = np_hori.argmax(axis=0)
    hori_max = horizontal_params[index_rho]
    backtorgb = draw_line(backtorgb, hori_max[0], hori_max[1])

    index_rho, index_theta = np_vert.argmin(axis=0)
    vert_min = vertical_params[index_rho]
    backtorgb = draw_line(backtorgb, vert_min[0], vert_min[1])

    index_rho, index_theta = np_hori.argmin(axis=0)
    hori_min = horizontal_params[index_rho]
    backtorgb = draw_line(backtorgb, hori_min[0], hori_min[1])

    cv2.imshow("lines for deskewing", backtorgb)
    return backtorgb



def compute_perspective_matrix(vertical_params, horizontal_params, h, w):
    np_vert = np.abs(np.array(vertical_params))
    np_hori = np.abs(np.array(horizontal_params))
    pers_matrix =  []
    #
    index_rho, index_theta = np_vert.argmax(axis=0)
    vert_max = vertical_params[index_rho]

    index_rho, index_theta = np_hori.argmax(axis=0)
    hori_max = horizontal_params[index_rho]

    index_rho, index_theta = np_vert.argmin(axis=0)
    vert_min = vertical_params[index_rho]

    index_rho, index_theta = np_hori.argmin(axis=0)
    hori_min = horizontal_params[index_rho]

    #print("vert_min:", vert_min, "vert_max:", vert_max)


    if abs(vert_max[0])-abs(vert_min[0]) < h/10 or abs(hori_max[0])-abs(hori_min[0]) < w/10:
        return pers_matrix

    p_left_top = compute_intersection(hori_min, vert_min)
    p_right_top = compute_intersection(hori_min, vert_max)
    p_left_bottom = compute_intersection(hori_max, vert_min)
    p_right_bottom = compute_intersection(hori_max, vert_max)


    pts1 = np.float32([p_left_top, p_right_top,p_left_bottom, p_right_bottom])
    pts2 = np.float32([p_left_top, [p_left_top[0] + dist(p_left_top, p_right_top), p_left_top[1]],
                [p_left_top[0], p_left_top[1] + dist(p_left_top, p_left_bottom)],
                [p_left_top[0] + dist(p_left_top, p_right_top), p_left_top[1] + dist(p_left_top, p_left_bottom)]])

    pers_matrix = cv2.getPerspectiveTransform(pts1, pts2)
    return pers_matrix


def get_perspective_transformed_im(gray):
    vertical_params = get_horizontal_vertical_lines(gray)
    h, w = gray.shape
    hori_line = get_text_line(gray)
    if hori_line[0] == 0 and hori_line[1] == 0:
        return gray, False
    hori_line_2 = [hori_line[0] + h/6, hori_line[1]]
    horizontal_params = []
    horizontal_params.append(hori_line)
    horizontal_params.append(hori_line_2)
    #print("vertical:", vertical_params)
    #print("hori:", horizontal_params)
    #check_line_selected(gray, vertical_params, horizontal_params)
    #print(vertical_params, horizontal_params)
    if len(vertical_params) > 1 and len(horizontal_params) > 1:
        h, w = gray.shape
        #check_line_selected(gray, vertical_params, horizontal_params)
        M = compute_perspective_matrix(vertical_params, horizontal_params, h, w)
        if M ==[]:
            return gray, False
        gray = cv2.warpPerspective(gray, M , (w, h))
    return gray, True


def resize_im(image):
    im_shape = image.shape
    #imgScale = 600/im_shape[1]
    imgScale = 800 / im_shape[1]
    newX,newY = image.shape[1]*imgScale, image.shape[0]*imgScale
    image = cv2.resize(image, (int(newX),int(newY)))
    return image

