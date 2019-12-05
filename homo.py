import cv2 as cv
import numpy as np
import random
import math

debug = True
homography_points = []

def add_homography_points(event, x, y, flags, param):
    global img, homography_points

    if event == cv.EVENT_LBUTTONDOWN:

        if homography_points.count >= 4:
            #Skip to next phase
            return

        
        print(x,', ',y)
        font = cv.FONT_HERSHEY_SIMPLEX
        strXY = str(x) + ', ' + str(y)
        cv.circle(img, (x,y), 10, (255,0,0))
        cv.imshow('Select 4 Points', img)

def loadImage(filename):
    img = cv.imread(filename)
    
    if debug:
        cv.imshow('Select 4 Points', img)

    return img

if __name__ == "__main__":
    global img
    for i in range(1, 4):
        filename = 'images/{}.png'.format(i)
        img = loadImage(filename)

        cv.setMouseCallback('Select 4 Points', add_homography_points)

        cv.waitKey(0)
    cv.destroyAllWindows()