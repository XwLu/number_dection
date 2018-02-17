# -*- coding: utf-8 -*-
#coding=utf-8

"""import the data"""

import model

isTrained = True
checkpoint_dir = '/home/luyifan/Project/tensorflow/number_detection/model'

if __name__ == "__main__":
  num_detector = model.NumberDetection()
  if not isTrained:
    num_detector.train(0.0001, 20000, 0.5)
    num_detector.saveWeights(checkpoint_dir)
  else:
    num_detector.loadWeights(checkpoint_dir)
    #num_detector.picturePredict('/home/luyifan/Project/tensorflow/number_detection/image/5.png')
    num_detector.cameraPredict()