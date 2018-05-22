import numpy as np
import cv2

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


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    cv2.imshow("edge",edged)
    # return the edged image
    return edged


def get_horizontal_vertical_lines(gray):
    vertical_params=[]
    horizontal_params = []
    gray = auto_canny(gray)
    #detect regions in gray scale image
    height, width = gray.shape
    lines = cv2.HoughLines(gray, rho=1, theta =np.pi/180, threshold = 160)

    if lines is not None:
        for line in lines:
            for rho, theta in line:
                if theta*180/np.pi < angle_threshold_line_detection or theta*180/np.pi > 180 - angle_threshold_line_detection:
                        vertical_params.append([rho, theta])
                if theta*180/np.pi > 90-angle_threshold_line_detection and theta*180/np.pi < 90 + angle_threshold_line_detection:
                    horizontal_params.append([rho, theta])
    return vertical_params, horizontal_params


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

    if abs(vert_max[0])-abs(vert_min[0]) < h/5 or abs(hori_max[0])-abs(hori_min[0]) < w/5:
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
    vertical_params, horizontal_params = get_horizontal_vertical_lines(gray)
    if len(vertical_params) > 1 and len(horizontal_params) > 1:
        h, w = gray.shape
        M = compute_perspective_matrix(vertical_params, horizontal_params, h, w)
        if M ==[]:
            return gray
        gray = cv2.warpPerspective(gray, M , (w, h))
    return gray

def resize_im(image):
    height, width, depth = image.shape
    imgScale = 600/width
    newX,newY = image.shape[1]*imgScale, image.shape[0]*imgScale
    image = cv2.resize(image, (int(newX),int(newY)))
    return image

def detect_shape(c):
    # initialize the shape name and approximate the contour
    shape = "unidentified"


    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)

    # if the shape is a triangle, it will have 3 vertices
    if len(approx) == 3:
        shape = "triangle"

    # if the shape has 4 vertices, it is either a square or
    # a rectangle
    elif len(approx) == 4:
        # compute the bounding box of the contour and use the
        # bounding box to compute the aspect ratio
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)

        # a square will have an aspect ratio that is approximately
        # equal to one, otherwise, the shape is a rectangle
        shape = "square" if ar >= 0.50 and ar <= 0.8 else "rectangle"


    # if the shape is a pentagon, it will have 5 vertices
    elif len(approx) == 5:
        shape = "pentagon"

    # otherwise, we assume the shape is a circle
    else:
        shape = "circle"

    # return the name of the shape
    return shape
