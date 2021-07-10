# Panorama Stitching from scratch 
# Inbuilt SIFT function is not used to find keypoints
import cv2 
import matplotlib.pyplot as plt
import numpy as np
import glob

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
    thresh = 0.1 * corner_list.max()
    result = []
    for j in range(0, corner_list.shape[0]):
        for i in range(0, corner_list.shape[1]):
            if(corner_list[j,i] > thresh):
                result.append([i,j])
                
    return result

def inputs():
    li = []
    count = 0
    for img in glob.glob("nature*.jpg"):
        read = cv2.imread(img)
        li.append(read)
        count += 1
        
    return li, count
    
def convert_into_keypts(corners):
    keypts = []
    for j in corners:
        temp = cv2.KeyPoint(int(j[0]), int(j[1]), 1)
        keypts.append(temp)
        
    return keypts

def SIFT(img):
    gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray12 = np.float32(gray1)
    sift = cv2.xfeatures2d.SIFT_create()
    result1 = find_harris(gray12, 3, 0.04)
    keypoints_1 = convert_into_keypts(result1)
    descriptors_1 = sift.compute(gray1, keypoints_1)[1]
    keypoints_1 = np.float32([kp.pt for kp in keypoints_1])
    
    return keypoints_1, descriptors_1

def good_ones(match):
    good = []
    for m,n in match:
        if m.distance < 0.75*n.distance:
            good.append((m.trainIdx, m.queryIdx))
            
    return good

def match(des1, des2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors_1,descriptors_2,k=2)
    good = good_ones(matches)
    
    return good

def stitch(img1, img2, right, row):
    left = np.zeros([row, img1.shape[1] + img2.shape[1], 3])
    left[0:row, 0:img1.shape[1]] = img1
    combine = [left, right]
    ans = np.max(combine, axis = 0)
    ans = ans.astype(np.uint8)
    
    return ans

images, tot = inputs()
final = images[0]
row = final.shape[0]
for i in range(1, tot):
    img = images[i]
    
    keypoints_1, descriptors_1 = SIFT(final)
    keypoints_2, descriptors_2 = SIFT(img)

    good = match(descriptors_1, descriptors_2)

    ptsA = np.float32([keypoints_1[i] for (_, i) in good])
    ptsB = np.float32([keypoints_2[i] for (i, _) in good])

    (H, status) = cv2.findHomography(ptsB, ptsA, cv2.RANSAC, 5.0)
    
    result = cv2.warpPerspective(img, H, (final.shape[1] + img.shape[1], row))
    
    ans = stitch(final, img, result, row)
    
    final = ans.copy()
    print(i)

cv2.imwrite('lets party.jpg', final)
cv2.imshow('answer', final)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




