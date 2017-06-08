from PIL import Image
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from ImageAI.gray_image import GreyImage
import time
import functools
from collections import Counter

i = Image.open('images/dot.png') # 8*8 image
iar = np.asarray(i) # 8 arrays corresponding to rows and 8 lines per each array corresponding to columns in those rows

# print(iar) # the output gives number 255 so the image was saved as a 256 bitmap
# threshold = GreyImage('images/numbers/y0.5.png').save_grey_image()

def createExamples():
    numberArrayExamples = open('numArEx.txt', 'a')
    numbersWeHave = range(0, 10)
    versionsWeHave = range(1,10)

    for eachNum in numbersWeHave:
        for eachVer in versionsWeHave:
            #print(str(eachNum)+'.'+str(eachVer))
            imgFilePath = 'images/numbers/'+str(eachNum)+'.'+str(eachVer)+'.png'
            ei = Image.open(imgFilePath)
            eiar = np.array(ei)
            eiar1 = str(eiar.tolist())

            lineToWrite = str(eachNum)+'::'+eiar1+'\n'
            numberArrayExamples.write(lineToWrite)

# createExamples()

def threshold(imageArray):
    ar=imageArray
    balanceAr = []
    newAr = ar
    total = 0

    for row in ar:
        for pix in row:
            avgNum = (pix[0] + pix[1] + pix[2])/3
            balanceAr.append(avgNum)
    for n in balanceAr:
        total += n
    balance = total/len(balanceAr)

    for eachRow in newAr:
        for eachPix in eachRow:
            avgNum = (eachPix[0] + eachPix[1] +eachPix[2])/3

            if avgNum > balance:
                eachPix[0] = 255
                eachPix[1] = 255
                eachPix[2] = 255
                eachPix[3] = 255
            else:
                eachPix[0] = 0
                eachPix[1] = 0
                eachPix[2] = 0
                eachPix[3] = 255

    return newAr

# function to take image and compare to database
def whatNumIsThis(filePath):
    matchedAr = []
    loadExamps = open('numArEx.txt', 'r').read()
    loadExamps = loadExamps.split('\n')

    i = Image.open(filePath)
    iar = np.array(i)
    iarl = iar.tolist()

    inQuestion = str(iarl)

    for eachExample in loadExamps:
        if len(eachExample) > 3: # not to have to worry about blank lines
            splitEx = eachExample.split('::')
            currentNum = splitEx[0]
            currentAr = splitEx[1]

            eachPixEx = currentAr.split('],')
            eachPixInQ = inQuestion.split('],')

            x = 0

            while x < len(eachPixEx):
                if eachPixEx[x] == eachPixInQ[x]:
                    matchedAr.append(int(currentNum))

                x+=1
    print(matchedAr)

    x = Counter(matchedAr) #counter goes through array and counts
    print(x)

    graphX = []
    graphY = []

    for eachThing in x:
        print(eachThing)
        graphX.append(eachThing)
        print(x[eachThing])
        graphY.append(x[eachThing])

    fig = plt.figure()
    ax1 = plt.subplot2grid((4,4), (0,0), rowspan=1, colspan=4)
    ax2 = plt.subplot2grid((4,4), (1,0), rowspan=3, colspan=4)

    ax1.imshow(iar)

    ax2.bar(graphX, graphY, align='center') #only wanna center when your x ticks are names or dates

    plt.ylim(300)

    xloc = plt.MaxNLocator(12)

    ax2.xaxis.set_major_locator(xloc)

    plt.show()



whatNumIsThis('/Users/mk/PycharmProjects/AI/ImageAI/ImageRecognition/images/numbers/test.png')



# i = Image.open('images/numbers/0.1.png')
# iar = np.array(i)
# threshold(iar)
# i2 = Image.open('images/numbers/y0.4.png')
# iar2 = np.array(i2)
# threshold(iar2)
#
#
# i3 = Image.open('images/numbers/y0.5.png')
# iar3 = np.array(i3)
# tresh = iar3
# threshold(tresh)
# print(tresh)
#
# i4 = Image.open('images/sentdex.png')
# iar4 = np.array(i4)
# threshold(iar4)
#
# fig = plt.figure()
# ax1 = plt.subplot2grid((8,6), (0,0), rowspan=4, colspan=3)
# ax2 = plt.subplot2grid((8,6), (4,0), rowspan=4, colspan=3)
# ax3 = plt.subplot2grid((8,6), (0,3), rowspan=4, colspan=3)
# ax4 = plt.subplot2grid((8,6), (4,3), rowspan=4, colspan=3)
#
# ax1.imshow(iar)
# ax2.imshow(iar2)
# ax3.imshow(iar3)
# ax4.imshow(iar4)
#
# plt.show()

# print('iar2', iar2)
# plt.imshow(iar2) # load image into it
# plt.show() # show the image
# #
# #
# i3 = Image.open('images/numbers/y0.5_grey.png')
# iar3 = np.asarray(i3)
# print('iar3', iar3)
# plt.imshow(iar3)  # load image into it
# plt.show()
# # # #1
#
# print(iar3)