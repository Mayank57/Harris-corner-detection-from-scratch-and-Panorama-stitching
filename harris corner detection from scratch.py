# Harris Corner detection from scratch 

import cv2 
import matplotlib.pyplot as plt
import numpy as np

def find_gradient(img):  
    x, y = img.shape
    dx = np.zeros((x, y))
    dy = np.zeros((x, y))
    for i in range(x):
        for j in range(y):
            if j == 0 :
                dx[i, j] = img[i, 1] - img[i, 0]
            elif j == y-1:
                dx[i, j] = img[i, y-1] - img[i, y-2]
            else:
                dx[i, j] = (img[i, j+1] - img[i, j-1]) / 2
    for i in range(y):
        for j in range(x):
            if j == 0:
                dy[j, i] = img[1, i] - img[0, i]
            elif j == x-1:
                dy[j, i] = img[x-1, i] - img[x-2, i]
            else:
                dy[j, i] = (img[j+1, i] - img[j-1, i]) / 2
        
    return dx,dy

def window(diff, a, b, c, d):
    out = diff[a : b, c : d]
    
    return out

def summation(a):
    out = np.sum(a)
    
    return out

def find_r(x, y, offset, k, Ixx, Ixy, Iyy):
    windowIxx = window(Ixx, y - offset, y + offset + 1, x - offset, x + offset + 1)
    windowIxy = window(Ixy, y - offset, y + offset + 1, x - offset, x + offset + 1)
    windowIyy = window(Iyy, y - offset, y + offset + 1, x - offset, x + offset + 1)

    Sxx = summation(windowIxx)
    Sxy = summation(windowIxy)
    Syy = summation(windowIyy)

    det = (np.multiply(Sxx, Syy)) - (np.square(Sxy))
    trace = Sxx + Syy

    r = det - k * (np.square(trace))
    
    return r

def my_harris_corners(input_img, window_size, k):
    
    corner_list = np.zeros(input_img.shape)
    
    offset = window_size // 2
    y_range = input_img.shape[0] - offset
    x_range = input_img.shape[1] - offset
    
    dx, dy = find_gradient(input_img)

    Ixx = np.square(dx)
    Ixy = np.multiply(dy, dx)
    Iyy = np.square(dy)
    
    for y in range(offset, y_range):
        for x in range(offset, x_range):          
            corner_list[y, x] = find_r(x, y, offset, k, Ixx, Ixy, Iyy)

    return corner_list

def find_harris(input_img, window_size, k):
    corner_list = my_harris_corners(input_img, window_size, k)
    corner_list = cv2.dilate(corner_list, None)
    thresh = 0.04 * corner_list.max()
    corner_img1 = img.copy()
    result = []
    for j in range(0, corner_list.shape[0]):
        for i in range(0, corner_list.shape[1]):
            if(corner_list[j,i] > thresh):
                result.append([i,j])
#                 cv2.circle(corner_img1, (i, j), 1, (0,255,0), 1)
    return result

img = cv2.imread("corner_test.png")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
sift = cv2.xfeatures2d.SIFT_create()
cornerList = find_harris(gray, 3, 0.04)

for px in cornerList:
    img[px[1],px[0],0] = 255
    img[px[1],px[0],1] = 0
    img[px[1],px[0],2] = 0

img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
cv2.imshow('answer', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

