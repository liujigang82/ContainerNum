import numpy as np
import cv2
import json
import os

def get_json_data():
    CONFIGFILE = os.getcwd()
    CONFIGFILE = CONFIGFILE + '/data/parameters.json'
    return json.load(open(CONFIGFILE))


def get_dict():
    alpha = "0123456789abcdefghijklmnopqrstuvwxyz"
    dict = {}
    counter = 0
    for i in range(len(alpha)):
        char = alpha[i]
        if i+counter == 11 or i+counter ==22 or i+counter == 33:
            counter += 1
        dict[char] = i + counter
    return dict

def get_encode_code(containerno):
    dict = get_dict()
    sum = 0
    for i in range(10):
        value = dict[containerno[i].lower()]
        sum += value * (2**i)
    return 0 if sum%11 == 10 else sum%11





