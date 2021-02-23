import cv2

import numpy as np

import matplotlib.pyplot as plt

 

 

 

origin_img = cv2.imread('lion.jpg', 0).astype(np.float32) # gray scale

origin_img = origin_img / 255

 

# DFT

fft = cv2.dft(origin_img, flags=cv2.DFT_COMPLEX_OUTPUT)

 

# spectrum visualization

shifted = np.fft.fftshift(fft, axes=[0,1])

magnitude = cv2.magnitude(shifted[:,:,0], shifted[:,:,1])

magnitude = np.log(magnitude)

 

plt.figure(figsize=(8,2))

plt.subplot(131)

plt.axis('off')

plt.title('origin_img')

plt.imshow(origin_img, cmap='gray')

plt.subplot(132)

plt.axis('off')

plt.title('magnitude')

plt.imshow(magnitude, cmap='gray')

plt.tight_layout()