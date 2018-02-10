import cv2
from PIL import Image
import glob
import numpy as np


def processImage(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray1 = cv2.cvtColor(img_gray, cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(img_gray1, 225, 255, cv2.THRESH_BINARY)
    thresh = cv2.bitwise_not(thresh)

    img1 = cv2.dilate(thresh, np.ones((4, 4)), iterations=10)
    img2 = cv2.erode(img1, np.ones((4, 4)), iterations=14)

    height, width, channels = img.shape
    img3, contours, hierarchy = cv2.findContours(img2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]

    hull = cv2.convexHull(cnt, returnPoints=False)
    newArray = (cnt[:, 0])
    niz = newArray[hull, :]

    ymin = niz[niz[:, 0][:, 1] == min(niz[:, 0][:, 1])][0] # gornja tacka
    xmin = niz[niz[:, 0][:, 0] == min(niz[:, 0][:, 0])][0] # leva u sredini
    xmax = niz[niz[:, 0][:, 0] == max(niz[:, 0][:, 0])][0] # desna u sredini
    ymaxarray = niz[niz[:, 0][:, 1] == max(niz[:, 0][:, 1])]
    ymax1 = ymaxarray[ymaxarray[:, 0][:, 0] == min(ymaxarray[:, 0][:, 0])][0] # donja leva tacka
    ymax2 = ymaxarray[ymaxarray[:, 0][:, 0] == max(ymaxarray[:, 0][:, 0])][0] # donja desna tacka

    tackeRet = np.matrix([[(xmax[0, 0]*100.0) / width, (xmax[0, 1]*100.0) / height], [(ymax2[0, 0]*100.0) / width, (ymax2[0, 1]*100.0) / height],
                       [(ymax1[0, 0]*100.0) / width, (ymax1[0, 1]*100.0) / height], [(xmin[0, 0]*100.0) / width, (xmin[0, 1]*100.0) / height],
                       [(ymin[0, 0]*100.0) / width, (ymin[0, 1]*100.0) / height]])

    return tackeRet


def writeToFile(coords,isVenomous, file):

    for coord in coords:
        file.write('%0.9lf,%0.9lf,' % (round(coord[0,:][0,0],9), round(coord[0,:][0,1],9)))
    if(isVenomous == 1):
        venomous = 1
        nonVenomous = 0
    else:
        venomous = 0
        nonVenomous = 1
    file.write('%d,%d \n' % (venomous, nonVenomous))


def loadImagesVenomous(file):
    for fileName in glob.glob('images/trainTest/Venomous Snakes/*.jpg'):
        img = Image.open(fileName)
        coords = processImage(np.asarray(img))
        writeToFile(coords, 1, file)


def loadImagesNonvenomous(file):
    for fileName in glob.glob('images/trainTest/NonvenomousSnakes/*.jpg'):
        img = Image.open(fileName)
        coords = processImage(np.asarray(img))
        writeToFile(coords, 0, file)
