import numpy as np
import cv2
 
 
imageA = cv2.imread('2.jpg') # 오른쪽 사진
imageB = cv2.imread('1.jpg') # 왼쪽 사진
 
grayA = cv2.cvtColor(imageA,cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB,cv2.COLOR_BGR2GRAY)
 
 
sift = cv2.xfeatures2d.SIFT_create()
kpA, desA = sift.detectAndCompute(grayA, None)
kpB, desB = sift.detectAndCompute(grayB, None)
 
 
bf = cv2.BFMatcher()
matches = bf.match(desA, desB)
 
 
sorted_matches = sorted(matches, key = lambda x : x.distance)
res = cv2.drawMatches(imageA, kpA, imageB, kpB, sorted_matches[:30], None, flags = 2)
 
 
src = np.float32([kpA[m.queryIdx].pt for m in matches]).reshape((-1, 1, 2))
dst = np.float32([kpB[m.trainIdx].pt for m in matches]).reshape((-1, 1, 2))
H, status = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
 
 
before2 = []
for x in range(imageA.shape[1], imageA.shape[1] * 2):
    for y in range(imageA.shape[0]):
        point = [x, y, 1]
        before2.append(point)
before2 = np.array(before2).transpose()
 
Hinv = np.linalg.inv(H)
 
after2 = np.matmul(Hinv, before2)
after2 = after2 / after2[2, :]
after2 = after2[:2, :]
after2 = np.round(after2, 0).astype(np.int)
 
height, width, _ = imageA.shape
result2 = np.zeros((height, width * 2, 3), dtype = np.uint8)
for pt1, pt2 in zip(before2[:2, :].transpose(), after2.transpose()):
    if pt2[1] >= height or pt2[0] >= width:
        continue
 
    if np.sum(pt2 < 0) >= 1:
        continue
    
    result2[pt1[1], pt1[0]] = imageA[pt2[1], pt2[0]]
result2[0: height, 0 : width] = imageB
cv2.imshow('result2', result2)
 
 
# cv2.warpPerspective( )
result3 = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
result3[0 : imageA.shape[0], 0 : imageB.shape[1]] = imageB
cv2.imshow('result3', result3) 
cv2.waitKey(0)
cv2.destroyAllWindows()