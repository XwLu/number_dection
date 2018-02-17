# -*- coding: utf-8 -*-
#coding=utf-8

import cv2
import numpy as np

cap = cv2.VideoCapture(0)
image = np.ndarray(shape=(784))
images = []

#while (1):
ret, frame = cap.read()

gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

img = cv2.resize(gray_img, dsize=(28, 28))

for i in range(28):
    for j in range(28):
        image[i*28+j] = img[i][j]/255.0

images = []
images.append(image)

images = np.asarray(images)
print(images)
cv2.imshow("capture", frame)
cv2.imwrite('test.png', frame)

    #if cv2.waitKey(1) & 0xFF == ord('q'):
     #   break