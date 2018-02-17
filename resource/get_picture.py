# -*- coding: utf-8 -*-
#coding=utf-8

import cv2
import numpy as np

"""通过以下方式，可以把一张28*28的图片读取进来，配置为网络的输入格式，输入到网络里"""

#image_path = '/home/luyifan/Project/number_detection/image/0.png'
image_path = '/home/luyifan/Project/tensorflow/number_detection/resource/test.png'

img_src = cv2.imread(image_path, 1)

gray_img = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)

"""先膨胀图像"""
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
dilated = cv2.dilate(thresh, kernel)
"""腐蚀图像"""
NpKernel = np.uint8(np.ones((3, 3)))
Nperoded = cv2.erode(dilated, NpKernel)

for i in range(len(Nperoded)):
    for j in range(len(Nperoded[0])):
        Nperoded[i][j] = 255 - Nperoded[i][j]

img = cv2.resize(Nperoded, dsize=(28, 28))

image = np.ndarray(shape=(784))

for i in range(len(img)):
    for j in range(len(img[0])):
        image[i*28+j] = img[i][j]/255.0

images = []
images.append(image)

images = np.asarray(images)

print(images)

cv2.imwrite("lat.png", Nperoded)

# image = np.reshape(image, [])