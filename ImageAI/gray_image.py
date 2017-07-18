import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from ImageAI.ImageRecognition.findextension import FindExtension

np.set_printoptions(threshold=np.nan)

"""This class just takes an image as input and saves a greyscale version of the image as originalimagename_grey.jpg
when you call GrayImage(path).save_grey_image()"""

class GreyImage:

    def __init__(self, path="pathasstring"):
        self.path = path
        self.image = cv2.imread(self.path)
        self.grey_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # self.save_name = FindExtension(self.path).add_name_before_extension('_grey')
        # self.save_name = FindExtension(self.path).add
    def save_grey_image(self):
        cv2.imwrite(self.save_name, self.grey_image)

# instance = GreyImage(path='/Users/mk/PycharmProjects/AI/ImageAI/CatDog/training_data/dogs/dog.148.jpg')
#
# cv2.imshow('greyclouds', instance.grey_image)
# #
# # instance.save_grey_image()
#
# fig = plt.figure()
#
# ax1 = plt.subplot2grid((8,6), (0,0), rowspan=4, colspan=3)
# # ax2 = plt.subplot2grid((8,6), (4,0), rowspan=4, colspan=3)
# # ax3 = plt.subplot2grid((8,6), (0,3), rowspan=4, colspan=3)
# # ax4 = plt.subplot2grid((8,6), (4,3), rowspan=4, colspan=3)
#
# iar = np.asarray(instance.grey_image)
#
# print(iar[0])
# #
# # saving = open("iar.txt", "w")
# # saving.write(str(iar))
# # saving.close()
# ax1.imshow(iar)
# # ax2.imshow(iar2)
# # ax3.imshow(iar3)
# # ax4.imshow(iar4)
# plt.show()
#
