import cv2
import numpy as np
from sklearn import linear_model
from textProcessing import isAlpha, containDigAlph, find_character_index
from utils import get_json_data


def is_point_in(point, pointList):
    for item in pointList:
        if point[0] == item[0] and point[1] == item[1]:
            return True
    return False


def get_image_patch(canvas, tesseract_data, result):
    t_boundary = 10
    for i in range(len(tesseract_data["text"])):
        itemList = tesseract_data["text"]
        text = [character for character in itemList[i] if character.isalpha()]
        text = "".join(item for item in text)
        if text in result and len(text) >= 2 and isAlpha(text):
            level_index = find_character_index(tesseract_data["level"], tesseract_data["level"][i])
            page_index = find_character_index(tesseract_data["level"], tesseract_data["level"][i])
            block_index = find_character_index(tesseract_data["block_num"], tesseract_data["block_num"][i])
            par_index = find_character_index(tesseract_data["par_num"], tesseract_data["par_num"][i])
            line_index = find_character_index(tesseract_data["line_num"], tesseract_data["line_num"][i])
            word_index = tesseract_data["word_num"][i]

            index = [i for i in level_index if
                     i in page_index and i in block_index and i in par_index and i in line_index]
            to_remove = []
            for i in range(len(index)):
                if not containDigAlph(tesseract_data["text"][index[i]]) or tesseract_data["word_num"][
                    index[i]] < word_index:
                    to_remove.append(index[i])
                    break
            for i in range(len(to_remove)):
                index.remove(to_remove[i])

            # index = index[word_index-1:len(index)]
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


def intersection(box1, box2):
    x = max(box1[0], box2[0])
    y = max(box1[1], box2[1])
    w = min(box1[0] + box1[2], box2[0] + box2[2]) - x
    h = min(box1[1] + box1[3], box2[1] + box2[3]) - y
    if w < 0 or h < 0: return (0, 0, 0, 0)
    return (x, y, w, h)


def not_inside(bbox, coords, method=0):
    if len(coords) == 0:
        return True
    else:
        for coord in coords:
            box = cv2.boundingRect(coord)
            # compare box and bbox
            if method == 1:
                intersects = intersection(box, bbox)
                if intersects != (0, 0, 0, 0):
                    return False
            else:
                if box[0] <= bbox[0] and box[1] <= bbox[1] \
                        and box[0] + box[2] >= bbox[0] + bbox[2] and box[1] + box[3] >= bbox[1] + bbox[3]:
                    return False
        return True


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


def contour_rec_ara(contour):
    bbox = cv2.boundingRect(contour)
    x, y, w, h = bbox
    return w * h


def detect(c):
    # initialize the shape name and approximate the contour
    shape = "unidentified"

    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)

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


def find_region_RANSAC(center_points, num_centers, y_position):
    ransac = linear_model.RANSACRegressor(residual_threshold=4)
    if len(center_points) > 0:
        X = np.array(center_points)[:, 0]
        Y = np.array(center_points)[:, 1]
        ransac.fit(X.reshape(-1, 1), Y)
        inlier_mask = ransac.inlier_mask_
        # print("inner", inlier_mask)
        tmp_centers = np.array(center_points)[inlier_mask]
        tmp_centers = tmp_centers[tmp_centers[:, 0].argsort()]

        if tmp_centers.shape[0] < 7:
            return num_centers

        y_centers = np.average(tmp_centers[:, 1])
        if y_centers < y_position:
            y_position = y_centers
            num_centers = tmp_centers

        outlier_mask = np.logical_not(inlier_mask)
        num_centers = find_region_RANSAC(np.array(center_points)[outlier_mask], num_centers, y_position)

    return num_centers


def get_binary_text_ROI(gray):
    parameters = get_json_data()
    threshold_width = float(parameters["threshold_width"]["value"])
    threshold_height = float(parameters["threshold_height"]["value"])

    mser = cv2.MSER_create()
    mser.setMaxArea(800)
    contours, bboxes = mser.detectRegions(gray)

    (height, width) = gray.shape[:2]
    coords = []
    contours = sorted(contours, key=contour_rec_ara, reverse=True)
    for c in contours:
        bbox = cv2.boundingRect(c)
        x, y, w, h = bbox
        if x > width * threshold_width and y < height * threshold_height and float(
                w / h) <= 1 and w < width / 15 and h < height / 15 and not_inside(bbox, coords):
            coords.append(c)

    canvas = np.zeros_like(gray)
    for cnt in coords:
        xx = cnt[:, 0]
        yy = cnt[:, 1]
        color = 255
        canvas[yy, xx] = color
    cv2.imshow("canvas1",canvas)

    im2, contours, hierarchy = cv2.findContours(canvas.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

    ### Use RANSAC to find the line of center points

    num_centers = find_region_RANSAC(center_points, "", 800)

    ### find the container no. position.Remove the forground not in num_centers.
    if len(num_centers) == 0:
        return canvas

    index = num_centers.shape[0] - 1
    for item in contour_info_list:
        if not is_point_in(item["center"], num_centers):
            rect = item["rect"]
            canvas[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]] = 0
        else:
            if np.array_equal(item["center"], num_centers[index]) and item["shape"] == "square":
                canvas = remove_rect(canvas, item["contour"])

    ##Get image patch
    x_left = num_centers[0][0]
    x_right = num_centers[num_centers.shape[0] - 1][0]

    tmp_centers = num_centers[num_centers[:, 1].argsort()]

    y_top = tmp_centers[0][1]
    y_bottom = tmp_centers[tmp_centers.shape[0] - 1][1]
    #print("num_centers:", x_left, x_right, y_top, y_bottom)
    #print(tmp_centers)
    canvas = canvas[y_top - 18:y_bottom+18, x_left-18:x_right+18]


    return canvas


def get_contour_list(gray):
    canvas = get_binary_text_ROI(gray)
    cv2.imshow("canvas", canvas)
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
            # print("unidentified")
            cv2.drawContours(backtorgb, [c], -1, (180, 180, 0), 2)
            # cv2.rectangle(backtorgb, (x, y), (x + w, y + h), (180, 180, 0), 2)
    cv2.imshow("Image", backtorgb)
