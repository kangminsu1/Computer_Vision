import cv2

import numpy as np

import matplotlib.pyplot as plt

import math

 

origin_img = cv2.imread('lion.jpg', 0).astype(np.float32) # gray scale

origin_img = origin_img / 255

 

# Garbor filter form

kernel = cv2.getGaborKernel((21,21), 5, 1, 10, 1, 0, cv2.CV_32F)

kernel /= math.sqrt((kernel * kernel).sum())

 
    
filtered = cv2.filter2D(origin_img, -1, kernel)

 

plt.figure(figsize=(8,3))

plt.subplot(131)

plt.axis('off')

plt.title('origin_img')

plt.imshow(origin_img, cmap='gray')

plt.subplot(132)

plt.axis('off')

plt.imshow(kernel, cmap='gray')

plt.title('kernel')

plt.subplot(133)

plt.axis('off')

plt.title('filtered')

plt.imshow(filtered, cmap='gray')

plt.tight_layout()

plt.show()