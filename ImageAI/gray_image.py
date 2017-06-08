import numpy as np
import cv2
import os
from ImageAI.ImageRecognition.findextension import FindExtension

"""This class just takes an image as input and saves a greyscale version of the image as originalimagename_grey.jpg
when you call GrayImage(path).save_grey_image()"""

class GreyImage:

    def __init__(self, path="pathasstring"):
        self.path = path
        self.image = cv2.imread(self.path)
        self.grey_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.save_name = FindExtension(self.path).add_name_before_extension('_grey')
    def save_grey_image(self):
        cv2.imwrite(self.save_name, self.grey_image)

# instance = GreyImage(path="clouds.jpg")
# cv2.imshow('greyclouds', instance.gray_image)
# instance.save_gray_image()

