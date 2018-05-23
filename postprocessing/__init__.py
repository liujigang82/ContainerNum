import cv2
import numpy as np
from textProcessing import  isAlpha, containDigAlph, find_character_index
from utils import get_json_data

def get_image_patch(canvas, tesseract_data, result):
    t_boundary = 10
    for i in range(len(tesseract_data["text"])):
        itemList = tesseract_data["text"]
        text = [character for character in itemList[i] if character.isalpha()]
        text = "".join(item for item in text)
        if text in result and len(text) >= 2 and isAlpha(text):
            print(text)
            level_index = find_character_index(tesseract_data["level"], tesseract_data["level"][i])
            page_index = find_character_index(tesseract_data["level"], tesseract_data["level"][i])
            block_index = find_character_index(tesseract_data["block_num"], tesseract_data["block_num"][i])
            par_index = find_character_index(tesseract_data["par_num"], tesseract_data["par_num"][i])
            line_index = find_character_index(tesseract_data["line_num"], tesseract_data["line_num"][i])
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


def intersection(box1,box2):
  x = max(box1[0], box2[0])
  y = max(box1[1], box2[1])
  w = min(box1[0]+box1[2], box2[0]+box2[2]) - x
  h = min(box1[1]+box1[3], box2[1]+box2[3]) - y
  if w<0 or h<0: return (0,0,0,0)
  return (x, y, w, h)


def not_inside(bbox, coords, method = 1):
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
                if box[0]<= bbox[0] and box[1] <= bbox[1] \
                    and box[0]+box[2]>=bbox[0]+bbox[2] and box[1]+box[3] >= bbox[1]+bbox[3]:
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
    return w*h

def detect(c):
    # initialize the shape name and approximate the contour
    shape = "unidentified"

    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)

    if cv2.isContourConvex(approx) and abs(cv2.contourArea(c))<200:
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


def get_binary_text_ROI(gray):
    parameters = get_json_data()
    threshold_width = float(parameters["threshold_width"]["value"])
    threshold_height = float(parameters["threshold_height"]["value"])

    mser = cv2.MSER_create()
    mser.setMaxArea(750)
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
            #cv2.rectangle(backtorgb, (x, y), (x + w, y + h), (0, 0, 255), 2)
        if shape == "rectangle":
            cv2.drawContours(backtorgb, [c], -1, (0, 255, 0), 2)
            #cv2.rectangle(backtorgb, (x, y), (x + w, y + h), (0, 255,), 2)
        if shape == "triangle":
            cv2.drawContours(backtorgb, [c], -1, (255, 0, 0), 2)
            #cv2.rectangle(backtorgb, (x, y), (x + w, y + h), (255, 0, 0), 2)
        if shape == "pentagon":
            cv2.drawContours(backtorgb, [c], -1, (255, 255, 0), 2)
            #cv2.rectangle(backtorgb, (x, y), (x + w, y + h), (255, 255, 0), 2)
        if shape == "circle":
            cv2.drawContours(backtorgb, [c], -1, (255, 0, 255), 2)
            #cv2.rectangle(backtorgb, (x, y), (x + w, y + h), (255, 0, 255), 2)
        if shape == "unidentified":
            #print("unidentified")
            cv2.drawContours(backtorgb, [c], -1, (180, 180, 0), 2)
            #cv2.rectangle(backtorgb, (x, y), (x + w, y + h), (180, 180, 0), 2)
    cv2.imshow("Image", backtorgb)