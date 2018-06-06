import cv2
import sys
sys.path.append('F:\\Projects\\ConainerNum\\ContainerNum')
import numpy as np
from matplotlib import pyplot as plt
from preprocessing  import get_perspective_transformed_im, resize_im #not_inside,
from sklearn import linear_model
from postprocessing import is_solid_box, not_inside, is_point_in, contour_rec_ara, find_region_RANSAC

# global para
threshold_width = 1/4
threshold_height = 1/2


def detect(c):
    # initialize the shape name and approximate the contour
    shape = "unidentified"

    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    if not cv2.isContourConvex(approx) or abs(cv2.contourArea(c)) < 50:
        print(cv2.isContourConvex(approx), abs(cv2.contourArea(c)))
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
    backtorgb = cv2.cvtColor(canvas,cv2.COLOR_GRAY2RGB)
    cv2.drawContours(backtorgb, [contour], -1, (0, 0, 0), 3)
    return cv2.cvtColor(backtorgb, cv2.COLOR_BGR2GRAY)

#82,
imageName = "img/img2/GESU.jpg"

img = cv2.imdecode(np.fromfile(imageName,dtype = np.uint8),-1)

img = resize_im(img)


cv2.imshow("image1", img)

'''
img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
'''
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = get_perspective_transformed_im(gray)

smoothed_img = cv2.GaussianBlur(gray, (3, 3), 0)
gray = cv2.addWeighted(gray, 1.5, smoothed_img, -0.5, 0)


#im2, contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)


mser = cv2.MSER_create()
#mser.setMinArea(50)
mser.setMaxArea(750)
contours, bboxes = mser.detectRegions(gray)


(height, width) = gray.shape[:2]
# loop over the contours
coords = []
contours = sorted(contours, key=contour_rec_ara, reverse=True)
print(len(contours))
for c in contours:
    bbox = cv2.boundingRect(c)
    x, y, w, h = bbox
    shape = detect(c)
    if not is_solid_box(w, h, c.shape[0]) and \
                    x > width * threshold_width and y < height * threshold_height and \
                    float(w / h) <= 1 and \
                    w < width/15 and h < height/15 and \
            not_inside(bbox, coords) :
        coords.append(c)


canvas = np.zeros_like(gray)
for cnt in coords:
    xx = cnt[:, 0]
    yy = cnt[:, 1]
    color = 255
    canvas[yy, xx] = color

cv2.imshow("canvas", canvas)
backtorgb = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
im2, contours, hierarchy = cv2.findContours(canvas.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

i = 0

contour_info_list = []
center_points = []
for cnt in contours:
    cnt = cv2.convexHull(cnt)
    shape = detect(cnt)
    x, y, w, h = cv2.boundingRect(cnt)
    if shape != "unidentified" :
        center_points.append([int(x+w/2), int(y+h/2)])
        contour_dict = {}
        contour_dict["rect"] = [x, y, w, h]
        contour_dict["contour"] = cnt
        contour_dict["shape"] = shape
        contour_dict["center"] = [int(x+w/2), int(y+h/2)]
        contour_info_list.append(contour_dict)

    cv2.rectangle(backtorgb, (x, y), (x+w, y+h), (255,0,0), 2)
    '''
    #cv2.drawContours(backtorgb, [cnt], -1, (0, 255, 255), 2)
    if shape == "square" :#or shape == "rectangle":
        cv2.drawContours(backtorgb, [cnt], -1, (0, 0, 255), 1)
    if shape == "rectangle":
        cv2.drawContours(backtorgb, [cnt], -1, (0, 255, 0), 2)
    if shape == "triangle":
        cv2.drawContours(backtorgb, [cnt], -1, (255, 0, 0), 2)
    if shape == "pentagon":
        cv2.drawContours(backtorgb, [cnt], -1, (255, 255, 0), 2)
    if shape == "circle":
        cv2.drawContours(backtorgb, [cnt], -1, (255, 0, 255), 2)
    '''
cv2.imshow("rect",backtorgb)
### Use RANSAC to find the line of center points

num_centers, line_x, line_y = find_region_RANSAC(center_points, "", 800, [], [])

if len(num_centers) > 1:
    ransac = linear_model.RANSACRegressor(residual_threshold=10)

    X = np.array(num_centers)[:, 0]
    Y = np.array(num_centers)[:, 1]

    ransac.fit(X.reshape(-1, 1), Y)
    inlier_mask = ransac.inlier_mask_

    ### find the container no. position.
    line_X = np.arange(X.min(), X.max())[:, np.newaxis]
    line_y_ransac = ransac.predict(line_X)

    plt.imshow(backtorgb)
    plt.plot(line_X, line_y_ransac, color='cornflowerblue', linewidth=2,
             label='RANSAC regressor')
    plt.xlabel("Input")
    plt.ylabel("Response")

### find the container no. position.Remove the forground not in num_centers.
print(num_centers)
if len(num_centers) != 0:

    index = num_centers.shape[0] -1
    for item in contour_info_list:
        if not is_point_in(item["center"], num_centers): #not in num_centers:
            rect = item["rect"]
            #print("not center:", np.array([item["center"]]).shape, num_centers.shape, np.array(item["center"]))
            canvas[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]] = 0
        else:
            cv2.drawContours(backtorgb, [item["contour"]], -1, (0, 0, 255), 1)
            if np.array_equal(item["center"], num_centers[index])and item["shape"] == "square":
                canvas = remove_rect(canvas, item["contour"])

cv2.imshow("remove redundant", canvas)


cv2.imshow("Image", backtorgb)
plt.show()

