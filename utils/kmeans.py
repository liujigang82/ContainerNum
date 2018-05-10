import cv2
from sklearn.cluster import KMeans
import numpy as np
from time import time
def kmeans(image):
    '''
    im_shape = image.shape
    if len(im_shape)==2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    '''
    #image = np.float32(image)
    t0 = time()
    k_means = KMeans(n_clusters=2)
    k_means.fit(np.array(image))
    values = k_means.cluster_centers_.squeeze()
    labels = k_means.labels_
    label_2 = k_means.predict(image)
    #print("kmeans:", values)
    #print("label:",labels)

    #
    #  convert to np.float32
    Z = image.reshape((-1, 3))
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((image.shape))
    cv2.imshow("kmeans", res2)
    cv2.imwrite("kmeans_4693.jpg",res2)
    cv2.imwrite("patch_4693.jpg", res2)
    return res2
