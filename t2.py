# 1. Only add your code inside the function (including newly improted packages).
#  You can design a new function and call the new function in the given functions.
# 2. For bonus: Give your own picturs. If you have N pictures, name your pictures such as ["t3_1.png", "t3_2.png", ..., "t3_N.png"], and put them inside the folder "images".
# 3. Not following the project guidelines will result in a 10% reduction in grades

import cv2
import numpy as np
# import matplotlib.pyplot as plt
import json

def dist(a, b):
    return np.sqrt(sum(np.square(a-b)))

def overlap(descrip_1, descrip_2, descrip_3, descrip_4):
    d = []
    dictionary_of_descrip = {1:descrip_1, 2:descrip_2, 3:descrip_3, 4:descrip_4}
    g=1
    while(g<5):
        dd1 = dictionary_of_descrip[g]
        o=1
        while(o<5):
            dd2 = dictionary_of_descrip[o]
            c = 0
            i=0
            while(i<len((dd1) - 150)):
                k=0
                while(k<len((dd2) - 150)):
                    descrip = dist(dd1[i], dd2[k])
                    if descrip < 100:
                        c = c + 1
                    k+=1
                i+=1
            o+=1
            if c < 2:
                d.append(0)
            else:
                d.append(1)
        g+=1
    overlap_arr = np.array(d)
    return overlap_arr

def stitch(imgmark, N=4, savepath=''): #For bonus: change your input(N=*) here as default if the number of your input pictures is not 4.
    "The output image should be saved in the savepath."
    "The intermediate overlap relation should be returned as NxN a one-hot(only contains 0 or 1) array."
    "Do NOT modify the code provided."
    imgpath = [f'./images/{imgmark}_{n}.png' for n in range(1, N+1)]
    imgs = []
    for ipath in imgpath:
        img = cv2.imread(ipath)
        imgs.append(img)
    "Start you code here"

    detector = cv2.xfeatures2d.SIFT_create(400)
    key_pt_1, descrip_1 = detector.detectAndCompute(imgs[0], None)
    key_pt_2, descrip_2 = detector.detectAndCompute(imgs[1], None)
    key_pt_3, descrip_3 = detector.detectAndCompute(imgs[2], None)
    key_pt_4, descrip_4 = detector.detectAndCompute(imgs[3], None)
    c = 0
    indexdes_1 = []
    indexdes_2 = []

    i=0
    while(i<len(descrip_1)):
        j=0
        while(j<len(descrip_2)):
            descrip = dist(descrip_1[i], descrip_2[j])
            if descrip >= 70:
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

    r=0
    while(r<len(indexdes_2)):
        keypt2_l.append(key_pt_2[indexdes_2[r]].pt)
        keypt2_2.append(key_pt_2[indexdes_2[r]])
        r+=1

    sourcepoints = np.float32(keypt1_l).reshape(-1, 1, 2)
    destinationpoints = np.float32(keypt2_l).reshape(-1, 1, 2)

    homo, stat = cv2.findHomography(sourcepoints, destinationpoints, cv2.RANSAC, 5.0)

    pts0 = np.array([[0, 0], [0, imgs[0].shape[0]], [imgs[0].shape[1], imgs[0].shape[0]], [imgs[1].shape[1], 0]], dtype=np.float32)
    pts0 = pts0.reshape((-1, 1, 2))
    pts1 = np.array([[0, 0], [0, imgs[1].shape[0]], [imgs[1].shape[1], imgs[1].shape[0]], [imgs[1].shape[1], 0]], dtype=np.float32)
    pts1 = pts1.reshape((-1, 1, 2))
    pts2 = cv2.perspectiveTransform(pts1, homo)

    pts = np.concatenate((pts0, pts2), axis=0)
    [minimum_x, minimum_y] = np.int32(pts.min(axis=0).ravel() - 0.6)
    [maximum_x, maximum_y] = np.int32(pts.max(axis=0).ravel() + 0.6)
    homo_translation = np.array([[1, 0, -minimum_x], [0, 1, -minimum_y], [0, 0, 1]])
    homof = homo_translation.dot(homo)

    op = cv2.warpPerspective(imgs[0], homof, (maximum_x - minimum_x, maximum_y - minimum_y))
    op[abs(minimum_y):imgs[1].shape[0] + abs(minimum_y), abs(minimum_x): imgs[1].shape[1] + abs(minimum_x)] = imgs[1]

    key_pt_5, descrip_5 = detector.detectAndCompute(op, None)
    indexdes_3 = []
    indexdes_4 = []

    i=0
    while(i<len(descrip_5)):
        k=0
        while(k<len(descrip_3)):
            descrip = dist(descrip_5[i], descrip_3[k])
            if descrip >= 150:
                pass
            else:
                c = c + 1
                indexdes_4.append(i)
                indexdes_3.append(k)
            k+=1
        i+=1

    keypt3_l = []
    keypt4_l = []

    i=0
    while(i<len(indexdes_4)):
        keypt4_l.append(key_pt_5[indexdes_4[i]].pt)
        i+=1

    r=0
    while(r<len(indexdes_3)):
        keypt3_l.append(key_pt_3[indexdes_3[r]].pt)
        r+=1

    sourcepoints = np.float32(keypt4_l).reshape(-1, 1, 2)
    destinationpoints = np.float32(keypt3_l).reshape(-1, 1, 2)

    homo, stat = cv2.findHomography(sourcepoints, destinationpoints, cv2.RANSAC, 5.0)
    pts0 = np.array([[0, 0], [0, op.shape[0]], [op.shape[1], op.shape[0]], [imgs[2].shape[1], 0]], dtype=np.float32)
    pts0 = pts0.reshape((-1, 1, 2))
    pts1 = np.array([[0, 0], [0, imgs[2].shape[0]], [imgs[2].shape[1], op.shape[0]], [imgs[2].shape[1], 0]], dtype=np.float32)
    pts1 = pts1.reshape((-1, 1, 2))
    pts2 = cv2.perspectiveTransform(pts1, homo)

    pts = np.concatenate((pts0, pts2), axis=0)
    [minimum_x, minimum_y] = np.int32(pts.min(axis=0).ravel() - 0.6)
    [maximum_x, maximum_y] = np.int32(pts.max(axis=0).ravel() + 0.6)
    homo_translation = np.array([[1, 0, -minimum_x], [0, 1, -minimum_y], [0, 0, 1]])
    homof = homo_translation.dot(homo)

    op1 = cv2.warpPerspective(op, homof, (maximum_x - minimum_x, maximum_y - minimum_y))
    op1[abs(minimum_y):abs(imgs[2].shape[0]) + abs(minimum_y), abs(minimum_x): abs(imgs[2].shape[1]) + abs(minimum_x)] = imgs[2]

    key_pt_6, descrip_6 = detector.detectAndCompute(op1, None)
    indexdes_5 = []
    indexdes_6 = []

    i=0
    while(i<len(descrip_6)):
        k=0
        while(k<len(descrip_4)):
            descrip = dist(descrip_6[i], descrip_4[k])
            if descrip >= 150:
                pass
            else:
                c = c + 1
                indexdes_6.append(i)
                indexdes_5.append(k)
            k+=1
        i+=1

    keypt5_l = []
    keypt6_l = []

    i=0
    while(i<len(indexdes_6)):
        keypt6_l.append(key_pt_6[indexdes_6[i]].pt)
        i+=1

    r=0
    while(r<len(indexdes_5)):
        keypt5_l.append(key_pt_4[indexdes_5[r]].pt)
        r+=1

    sourcepoints = np.float32(keypt6_l).reshape(-1, 1, 2)
    destinationpoints = np.float32(keypt5_l).reshape(-1, 1, 2)

    homo, stat = cv2.findHomography(sourcepoints, destinationpoints, cv2.RANSAC, 5.0)
    pts0 = np.array([[0, 0], [0, op1.shape[0]], [op1.shape[1], op1.shape[0]], [imgs[3].shape[1], 0]], dtype=np.float32)
    pts0 = pts0.reshape((-1, 1, 2))
    pts1 = np.array([[0, 0], [0, imgs[3].shape[0]], [imgs[3].shape[1], op1.shape[0]], [imgs[3].shape[1], 0]], dtype=np.float32)
    pts1 = pts1.reshape((-1, 1, 2))
    pts2 = cv2.perspectiveTransform(pts1, homo)

    pts = np.concatenate((pts0, pts2), axis=0)
    [minimum_x, minimum_y] = np.int32(pts.min(axis=0).ravel() - 0.6)
    [maximum_x, maximum_y] = np.int32(pts.max(axis=0).ravel() + 0.6)
    homo_translation = np.array([[1, 0, -minimum_x], [0, 1, -minimum_y], [0, 0, 1]])
    homof = homo_translation.dot(homo)

    op2 = cv2.warpPerspective(op1, homof, (maximum_x - minimum_x, maximum_y - minimum_y))
    op2[abs(minimum_y):imgs[3].shape[0] + abs(minimum_y), abs(minimum_x): imgs[3].shape[1] + abs(minimum_x)] = imgs[3]

    cv2.imwrite(savepath, op2)

    overlap_arr = overlap(descrip_1, descrip_2, descrip_3, descrip_4)
    return overlap_arr

if __name__ == "__main__":
    #task2
    overlap_arr = stitch('t2', N=4, savepath='task2.png')
    with open('t2_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr.tolist(), outfile)

    # bonus
    # overlap_arr2 = stitch('t3', N=4, savepath='task3.png')
    # with open('t3_overlap.txt', 'w') as outfile:
    #     json.dump(overlap_arr2.tolist(), outfile)