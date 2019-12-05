import cv2 as cv
import numpy as np
import random
import math

debug = True

class Homo:
    homography_points = []
    homography = []
    img = []

    def __init__(self, img):
        self.img = img

    def calculate_homography(self):
        rw_points = np.array([[11,5.5],[29.32,5.5],[12.85,16.5],[27.47,16.5]])

        self.homography, status = cv.findHomography(np.array(self.homography_points), rw_points)

    def drawGoalCircle(self):
        rw_goalPoint = np.array([[20.16,0]]).reshape(-1,1,2)

        img_goalPoint = cv.perspectiveTransform(np.array(rw_goalPoint), np.linalg.inv(self.homography))
        tuple_goalPoint = (int(img_goalPoint[0][0][0]),int(img_goalPoint[0][0][1]))  

        cv.circle(img,tuple_goalPoint,10,(0,255,0))  
        cv.imshow('Select 4 Points', self.img) 

    def add_homography_points(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            if len(self.homography_points) < 4:
                print(x,', ',y)
                cv.circle(img, (x,y), 10, (255,0,0))
                cv.imshow('Select 4 Points', self.img)
                self.homography_points.append([x,y])

            if len(self.homography_points) >= 4:
                self.calculate_homography()
                self.drawGoalCircle()

def loadImage(filename):
    img = cv.imread(filename)
    
    if debug:
        cv.imshow('Select 4 Points', img)

    return img

if __name__ == "__main__":
    for i in range(1, 4):
        filename = 'images/{}.png'.format(i)
        img = loadImage(filename)

        homo = Homo(img)

        cv.setMouseCallback('Select 4 Points', homo.add_homography_points)

        cv.waitKey(0)
    cv.destroyAllWindows()