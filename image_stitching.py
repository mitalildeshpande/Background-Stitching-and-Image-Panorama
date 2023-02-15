#Only add your code inside the function (including newly improted packages)
# You can design a new function and call the new function in the given functions. 
# Not following the project guidelines will result in a 10% reduction in grades

import cv2
import numpy as np
# import matplotlib.pyplot as plt


def stitch_background(img1, img2, savepath=''):
    "The output image should be saved in the savepath."
    "Do NOT modify the code provided."

    detector = cv2.xfeatures2d.SIFT_create(910)
    key_pt_1, descrip_1 = detector.detectAndCompute(img1, None)
    key_pt_2, descrip_2 = detector.detectAndCompute(img2, None)
    c = 0

    indexdes_1 = []
    indexdes_2 = []

    i=0
    while(i<len(descrip_1)):
        j=0
        while(j<len(descrip_2)):
            discrip = np.sqrt(sum(np.square(descrip_1[i] - descrip_2[j])))
            if discrip >= 100:
                pass
            else:
                c = c + 1
                indexdes_1.append(i)
                indexdes_2.append(j)
            j+=1
        i+=1

    keypt1_1 = []
    keypt2_2 = []

    keypt1_l = []
    keypt2_l = []

    i=0
    while(i<len(indexdes_1)):
        keypt1_l.append(key_pt_1[indexdes_1[i]].pt)
        keypt1_1.append(key_pt_1[indexdes_1[i]])
        i+=1

    k=0
    while(k<len(indexdes_2)):
        keypt2_l.append(key_pt_2[indexdes_2[k]].pt)
        keypt2_2.append(key_pt_2[indexdes_2[k]])
        k+=1

    sourcepoints = np.float32(keypt1_l).reshape(-1, 1, 2)
    destinationpoints = np.float32(keypt2_l).reshape(-1, 1, 2)

    homo, stat = cv2.findHomography(sourcepoints, destinationpoints, cv2.RANSAC, 5.0)
    distance = cv2.warpPerspective(img1, homo, ((img1.shape[1] + img2.shape[1]), img1.shape[0] + img2.shape[0]))

    for i in range(0, img2.shape[0]):
        for k in range(0, img2.shape[1]):
            if np.sum(distance[i][k]) < 0:
                distance[i][k] = img2[i][k]
            else:
                if np.sum(distance[i][k]) < np.sum(img2[i][k]):
                    distance[i][k] = img2[i][k]
                else:
                    distance[i][k] = distance[i][k]

    cv2.imwrite(savepath, distance)

    return
if __name__ == "__main__":
    img1 = cv2.imread('./images/t1_1.png')
    img2 = cv2.imread('./images/t1_2.png')
    savepath = 'task1.png'
    stitch_background(img1, img2, savepath=savepath)
