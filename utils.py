import numpy as np
import cv2
import json
import os

def get_json_data():
    CONFIGFILE = os.getcwd()
    CONFIGFILE = CONFIGFILE + '/data/parameters.json'
    return json.load(open(CONFIGFILE))








