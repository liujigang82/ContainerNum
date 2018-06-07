import cv2
import numpy as np
from sklearn import linear_model
from textProcessing import isAlpha, containDigAlph, find_character_index
from utils import get_json_data
import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'Tesseract-OCR/tesseract'


def is_point_in(point, pointList):
    for item in pointList:
        if point[0] == item[0] and point[1] == item[1]:
            return True
    return False


def union_rect(box1, box2):
    x = min(box1[0], box2[0])
    y = min(box1[1], box2[1])
    w = max(box1[0] + box1[2], box2[0] + box2[2]) - x
    h = max(box1[1] + box1[3], box2[1] + box2[3]) - y
    return (x, y, w, h)


def intersection(box1, box2):
    x = max(box1[0], box2[0])
    y = max(box1[1], box2[1])
    w = min(box1[0] + box1[2], box2[0] + box2[2]) - x
    h = min(box1[1] + box1[3], box2[1] + box2[3]) - y
    if w < 0 or h < 0: return (0, 0, 0, 0)
    return (x, y, w, h)


def is_overlapping(box1, box2, intersec):
    area_box1 = float(box1[2] * box1[3])
    area_box2 = float(box2[2] * box2[3])
    area_intersect = float(intersec[2] * intersec[3])

    if area_intersect == 0:
        return False
    elif area_intersect / area_box1 > 0.5 or area_intersect / area_box2 > 0.5:
        return True
    return False


def is_inside(bbox, coords, method = 1):
    if len(coords) == 0:
        return False
    else:
        for coord in coords:
            box = cv2.boundingRect(coord)
            # compare box and bbox
            if method == 1:  # by intersection.
                intersects = intersection(box, bbox)
                if is_overlapping(box, bbox, intersects):
                    return True
            else:  # by check inside.
                if box[0] <= bbox[0] and box[1] <= bbox[1] \
                        and box[0] + box[2] >= bbox[0] + bbox[2] and box[1] + box[3] >= bbox[1] + bbox[3]:
                    return True
        return False


def calculateAngle(binary):
    coords = np.column_stack(np.where(binary > 0))
    angle = cv2.minAreaRect(coords)[-1]

    # the `cv2.minAreaRect` function returns values in the
    # range [-90, 0); as the rectangle rotates clockwise the
    # returned angle trends to 0 -- in this special case we
    # need to add 90 degrees to the angle
    if angle < -45:
        angle = -(90 + angle)

    # otherwise, just take the inverse of the angle to make
    # it positive
    else:
        angle = -angle

    (h, w) = binary.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(binary, M, (w, h),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


def contour_rec_ara(contour, method = 0):
    if method == 1:
        cnt = cv2.convexHull(contour)
        return cv2.contourArea(cnt)
    else:
        bbox = cv2.boundingRect(contour)
        x, y, w, h = bbox
    return w * h


def detect(c):
    # initialize the shape name and approximate the contour
    shape = "unidentified"

    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    if not cv2.isContourConvex(approx) or abs(cv2.contourArea(c)) < 50:
        return shape
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
        shape = "square" if ar >= 0.50 and ar <= 0.9 else "rectangle"

    # if the shape is a pentagon, it will have 5 vertices
    elif len(approx) == 5:
        shape = "pentagon"

    # otherwise, we assume the shape is a circle
    else:
        shape = "circle"

    # return the name of the shape
    return shape


def remove_rect(canvas, contour):
    backtorgb = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(backtorgb, [contour], -1, (0, 0, 0), 3)
    return cv2.cvtColor(backtorgb, cv2.COLOR_BGR2GRAY)


def find_region_RANSAC(center_points, num_centers, y_position, line_x, line_y):
    ransac = linear_model.RANSACRegressor(residual_threshold=4)

    if len(center_points) > 1:
        X = np.array(center_points)[:, 0]
        Y = np.array(center_points)[:, 1]
        ransac.fit(X.reshape(-1, 1), Y)
        inlier_mask = ransac.inlier_mask_

        tmp_centers = np.array(center_points)[inlier_mask]
        tmp_centers = tmp_centers[tmp_centers[:, 0].argsort()]
        if tmp_centers.shape[0] < 7:
            return num_centers, line_x, line_y

        y_centers = np.average(tmp_centers[:, 1])
        if y_centers < y_position:
            y_position = y_centers
            num_centers = tmp_centers
            line_x = np.array([X.min(), X.max()])
            line_x_forPre = np.array([X.min(), X.max()])[:, np.newaxis]
            line_y = ransac.predict(line_x_forPre)
        outlier_mask = np.logical_not(inlier_mask)
        num_centers, line_x, line_y = find_region_RANSAC(np.array(center_points)[outlier_mask], num_centers, y_position,
                                                         line_x, line_y)
    return num_centers, line_x, line_y

def is_solid_box(cnt, method = 0):
    x, y, w, h = cv2.boundingRect(cnt)
    cnt_hull = cv2.convexHull(cnt)
    #print("~~~",cv2.contourArea(cnt),  cv2.contourArea(cnt_1), w*h )
    if float(h)/float(w) > 3:
        return False
    if method == 1:
        if float(cnt.shape[0]) / float(w*h) < 0.65:
            return  False
    else:
        area = cv2.contourArea(cnt_hull)
        if area == 0:
            return True
        if float(cnt.shape[0]) / float(area) < 0.90:
            return False

    return True

def get_contour(gray):
    parameters = get_json_data()
    threshold_width = float(parameters["threshold_width"]["value"])
    threshold_height = float(parameters["threshold_height"]["value"])

    mser = cv2.MSER_create()
    #mser.setMaxArea(750)
    contours, bboxes = mser.detectRegions(gray)
    (height, width) = gray.shape[:2]
    coords = []
    contours = sorted(contours, key=contour_rec_ara, reverse=True)
    for c in contours:
        bbox = cv2.boundingRect(c)
        x, y, w, h = bbox
        if not is_solid_box(c) and \
                        x > width * threshold_width and y < height * threshold_height and \
                        float(w / h) <= 1 and \
                        w < width / 15 and h < height / 15 and \
                not is_inside(bbox, coords):
            coords.append(c)

    canvas = np.zeros_like(gray)
    for cnt in coords:
        xx = cnt[:, 0]
        yy = cnt[:, 1]
        color = 255
        canvas[yy, xx] = color
    #cv2.imshow("canvas_ori",canvas)
    im2, contours, hierarchy = cv2.findContours(canvas.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, canvas


def get_text_line(gray):
    contours, canvas = get_contour(gray)
    center_points = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cnt = cv2.convexHull(cnt)
        shape = detect(cnt)
        if shape != "unidentified":
            center_points.append([int(x + w / 2), int(y + h / 2)])
    ### Use RANSAC to find the line of center points
    num_centers, line_X, line_Y = find_region_RANSAC(center_points, "", 800, [], [])
    #print("Line:", line_X, line_Y)
    #print("number cernter:", num_centers)
    if len(line_X) < 2:
        return 0, 0

    if line_Y[0] == line_Y[1]:
        theta = np.pi/2
        rho = line_Y[0]
    else:
        theta = np.arctan((line_X[1] - line_X[0]) / (line_Y[0] - line_Y[1]))
        if theta < 0:

            theta = theta + np.pi
        rho = line_X[0] * np.cos(theta) + line_Y[0] * np.sin(theta)
    return [rho - 20, theta]


def get_text_contour(gray):
    contours, canvas = get_contour(gray)
    contour_info_list = []
    center_points = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cnt = cv2.convexHull(cnt)
        shape = detect(cnt)
        if shape != "unidentified":
            center_points.append([int(x + w / 2), int(y + h / 2)])
            contour_dict = {}
            contour_dict["rect"] = [x, y, w, h]
            contour_dict["contour"] = cnt
            contour_dict["shape"] = shape
            contour_dict["center"] = [int(x + w / 2), int(y + h / 2)]
            contour_info_list.append(contour_dict)
        else:
            canvas[y:y + h, x:x + w] = 0
    ### Use RANSAC to find the line of center points
    num_centers, line_X, line_Y = find_region_RANSAC(center_points, "", 800, [], [])
    return num_centers, canvas, contour_info_list


def get_binary_text_ROI(gray):
    num_centers, canvas, contour_info_list = get_text_contour(gray)
    ### find the container no. position.Remove the forground not in num_centers.

    if len(num_centers) == 0:
        return canvas

    index = num_centers.shape[0] - 1
    pts = []
    for item in contour_info_list:
        rect = item["rect"]
        if not is_point_in(item["center"], num_centers):
            canvas[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]] = 0
        else:
            pts.append([rect[0], rect[1]])
            pts.append([rect[0]+rect[2], rect[1]+rect[3]])
            if np.array_equal(item["center"], num_centers[index]) and item["shape"] == "square":
                canvas = remove_rect(canvas, item["contour"])

    ##Get image patch
    if len(pts) == 0:
        return canvas

    max_x, max_y = max(pts)
    min_x, min_y = min(pts)

    min_y = min_y - 10 if min_y - 10 >=0 else 0
    min_x = min_x - 10 if min_x - 10 >=0 else 0
    canvas = canvas[min_y:max_y+10, min_x:max_x+10]
    return canvas


def get_contour_list(gray):
    canvas = get_binary_text_ROI(gray)
    #cv2.imshow("canvas", canvas)
    im2, contours, hierarchy = cv2.findContours(canvas.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    backtorgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        c = cv2.convexHull(c)
        shape = detect(c)

        cv2.drawContours(backtorgb, [c], -1, (0, 255, 255), 2)
        if shape == "square":  # or shape == "rectangle":
            cv2.drawContours(backtorgb, [c], -1, (0, 0, 255), 2)
            # cv2.rectangle(backtorgb, (x, y), (x + w, y + h), (0, 0, 255), 2)
        if shape == "rectangle":
            cv2.drawContours(backtorgb, [c], -1, (0, 255, 0), 2)
            # cv2.rectangle(backtorgb, (x, y), (x + w, y + h), (0, 255,), 2)
        if shape == "triangle":
            cv2.drawContours(backtorgb, [c], -1, (255, 0, 0), 2)
            # cv2.rectangle(backtorgb, (x, y), (x + w, y + h), (255, 0, 0), 2)
        if shape == "pentagon":
            cv2.drawContours(backtorgb, [c], -1, (255, 255, 0), 2)
            # cv2.rectangle(backtorgb, (x, y), (x + w, y + h), (255, 255, 0), 2)
        if shape == "circle":
            cv2.drawContours(backtorgb, [c], -1, (255, 0, 255), 2)
            # cv2.rectangle(backtorgb, (x, y), (x + w, y + h), (255, 0, 255), 2)
        if shape == "unidentified":
            cv2.drawContours(backtorgb, [c], -1, (180, 180, 0), 2)
            # cv2.rectangle(backtorgb, (x, y), (x + w, y + h), (180, 180, 0), 2)
    cv2.imshow("Image", backtorgb)
