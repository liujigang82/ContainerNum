import numpy as np
import cv2


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


def find_index_word(str):
    length = 0
    word = ""
    sub_str_list = str.split(" ")
    for sub in sub_str_list:
        if is_text(sub) and len(sub) > length:
            length = len(sub)
            word = sub
    if length > 0:
        index = str.index(word)
        str = str[index:len(str)]
    return str, len(word)-1

def get_image_patch(canvas, tesseract_data, result):
    t_boundary = 7
    for i in range(len(tesseract_data["text"])):
        itemList = tesseract_data["text"]
        text = [character for character in itemList[i] if character.isalpha()]
        text = "".join(item for item in text)

        if text in result and len(text) >= 2 and is_text(text):
            level_index = myfind(tesseract_data["level"], tesseract_data["level"][i])
            page_index = myfind(tesseract_data["level"], tesseract_data["level"][i])
            block_index = myfind(tesseract_data["block_num"], tesseract_data["block_num"][i])
            par_index = myfind(tesseract_data["par_num"], tesseract_data["par_num"][i])
            line_index = myfind(tesseract_data["line_num"], tesseract_data["line_num"][i])


            index = [i for i in level_index if
                     i in page_index and i in block_index and i in par_index and i in line_index]

            to_remove = []
            for i in range(len(index)):
                if not containDigAlph(tesseract_data["text"][index[i]]):
                    to_remove.append(index[i])
                    break
            for i in range(len(to_remove)):
                index.remove(to_remove[i])

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