import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
np.set_printoptions(threshold=np.nan)
np.set_printoptions(precision=3)
import csv


image_path = '/Users/mk/PycharmProjects/AI/TFPOETS/Test/Grey/4x28_grey_0013.jpg'
# image_size = 128


i = Image.open(image_path)
i2 = cv2.imread(image_path)
# i2 = cv2.resize(i2, (image_size, image_size), cv2.INTER_LINEAR)
# greyi2 = cv2.cvtColor(i2, cv2.COLOR_BGR2GRAY)
iar = np.array(i)  # This one has one value per pixel <--- Better one
iar2 = np.array(i2) #This one has 3 identical values per pixel
original_image = np.array(i)
iar_mod = iar

# iar_mod = cv2.cvtColor(iar_mod, cv2.COLOR_BGR2GRAY)

threshold = 150

for undex in range(len(iar_mod)):
    for index, value in enumerate(iar_mod[undex]):
        if value > threshold:
            iar_mod[undex][index]=255
            # print('changed value to 255')
        elif value <= threshold:
            iar_mod[undex][index]=0
            # print('changed value to 0')

print(iar_mod)

# This part overlays a 5x5 ones matrix divided by 25 on the image and adds the pixels below this matrix (image)
# together and replaces the middle pixel with the average value
kernel = np.ones((5,5), np.float32)/25
kernel = kernel * 2
dst = cv2.filter2D(iar_mod,-1,kernel)


edges = cv2.Canny(dst,60,210, L2gradient=True)




plt.subplot(121),plt.imshow(i,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(dst,cmap = 'gray')
plt.title('Changed Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(edges, cmap = 'gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()