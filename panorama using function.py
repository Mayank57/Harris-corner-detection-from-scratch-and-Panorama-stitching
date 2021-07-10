#Panorama stitching using inbuilt SIFT function
import cv2 
import matplotlib.pyplot as plt
import numpy as np
import glob
cv_img = []
for img in glob.glob("nature*.jpg"):
    n= cv2.imread(img)
    cv_img.append(n)
temp = cv_img[0]
n = len(cv_img)
column = temp.shape[1]
for i in range(1, n):
    img = cv_img[i]
    gray1 = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints_1, descriptors_1 = sift.detectAndCompute(gray1,None)
    keypoints_1 = np.float32([kp.pt for kp in keypoints_1])
    gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#keypoints
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints_2, descriptors_2 = sift.detectAndCompute(gray2,None)
    keypoints_2 = np.float32([kp.pt for kp in keypoints_2])

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors_1,descriptors_2,k=2)
    # Apply ratio test
    good = []
    good1 = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append((m.trainIdx, m.queryIdx))
            good1.append([m])
    ptsA = np.float32([keypoints_1[i] for (_, i) in good])
    ptsB = np.float32([keypoints_2[i] for (i, _) in good])
    (H, status) = cv2.findHomography(ptsB, ptsA, cv2.RANSAC, 5.0)
    H_inv = np.linalg.inv(H)
    result1 = cv2.warpPerspective(img, H, (temp.shape[1] + img.shape[1], temp.shape[0]))
    cv2.imshow('imgi', result1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#     ans = np.zeros([temp.shape[0], temp.shape[1]+img.shape[1], 3], np.uint8)
#     print(img.shape, result.shape)
    result2 = np.zeros([temp.shape[0], temp.shape[1] + img.shape[1], 3])
    result2[0:temp.shape[0], 0:temp.shape[1]] = temp
#     temp = np.concatenate([temp, np.zeros(img.shape[0], img.shape[1], 3)])
#     print(result2.shape, result1.shape)
    l = [result1, result2]
    ans1 = np.max(l, axis = 0)
    ans1 = ans1.astype(np.uint8)
    temp = ans1.copy()
    
  
cv2.waitKey(0)
cv2.destroyAllWindows()

