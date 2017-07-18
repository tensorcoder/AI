import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('coc-0231 copy.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img2 = img.copy()
template = cv2.imread('coc-0250.jpg')
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

w, h = template.shape[::-1]


# All the 6 methods for comparison in a list
# methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
#             'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

methods = ['cv2.TM_CCOEFF']

for meth in methods:
    img = img2.copy()
    method = eval(meth)

    # Apply template Matching
    res = cv2.matchTemplate(img,template,method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    # cv2.rectangle(img,top_left, bottom_right, 255, 2)
    cv2.rectangle(img, top_left, bottom_right, 255, 2, 0, 0)
    # print('top left: ', top_left, 'bottom right: ', bottom_right)
    location = top_left[0], top_left[1], w, h
    print(location)
    plt.subplot(121),plt.imshow(res,cmap = 'gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img,cmap = 'gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)

    plt.show()

#
# img = cv2.imread('coc-0231 copy.jpg')
# mask = np.zeros(img.shape[:2],np.uint8)
#
# bgdModel = np.zeros((1,65),np.float64)
# fgdModel = np.zeros((1,65),np.float64)
#
# rect = location
#
# cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
#
# mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
# img = img*mask2[:,:,np.newaxis]
#
# plt.imshow(img),plt.colorbar(),plt.show()