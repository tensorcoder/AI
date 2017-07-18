import os
# from ImageAI.gray_image import GreyImage
import cv2
from ImageAI.ImageRecognition.findextension import FindExtension
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

pathasstring = '/Users/mk/PycharmProjects/AI/TFPOETS/Test/Grey'
os.mkdir('/Users/mk/PycharmProjects/AI/TFPOETS/Test/Changed')

directory = os.fsdecode(pathasstring)
counter = 0


def counter_helper(counter):
    if len(str(counter)) < 4:
        add_how_many = 4 - len(str(counter))
        if add_how_many == 2:
            return "00" + str(counter)
        elif add_how_many == 3:
            return "000" + str(counter)
        elif add_how_many == 1:
            return "0" + str(counter)
        else:
            print('Error Occured In counter_helper!!')
    else:
        return str(counter)


for filename in os.listdir(directory):
    if filename.endswith('.jpg'):
        fullpath = os.path.join(directory, filename)
        print(fullpath)
        Img = cv2.imread(fullpath)
        #Place to change each image however you want.
        changed_image = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)




        #CV manipulations
        counter += 1
        indexingfeature = counter_helper(counter)
        savename = '4x28_changed_' + indexingfeature + '.jpg'
        os.chdir('/Users/mk/PycharmProjects/AI/TFPOETS/Test/Changed')
        cv2.imwrite(savename, changed_image)
        print(savename)
        continue
    else:
        continue
