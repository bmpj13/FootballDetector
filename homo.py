import cv2 as cv
import numpy as np
import random
import math

debug = True

class Homo:
    homography_points = []
    homography = []
    img = []

    QUALITY_MULTIPLIER = 30
    GOAL_CENTER = 20.16 * QUALITY_MULTIPLIER

    def __init__(self, img):
        self.img = img

    def calculate_homography(self):
        rw_points = np.array([[11,5.5],[29.32,5.5],[12.85,16.5],[27.47,16.5]],dtype=np.float32) * self.QUALITY_MULTIPLIER

        self.homography, status = cv.findHomography(np.array(self.homography_points), rw_points)

    def click_event(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            if len(self.homography_points) < 4:
                print(x,', ',y)
                cv.circle(img, (x,y), 10, (255,0,0))
                cv.imshow('Select 4 Points', self.img)
                self.homography_points.append([x,y])
            
            if len(self.homography_points) >= 4:
                if len(self.homography) == 0:
                    self.calculate_homography()
                    return False
                return True
            else:
                return False

class GoalArrowDistance(Homo):
    def click_event(self, event, x, y, flags, param):
        if (super().click_event(event,x,y,flags,param)):
            img_ballPoint = np.array([[float(x),float(y)]]).reshape(-1,1,2)
            rw_ballPoint = cv.perspectiveTransform(np.array(img_ballPoint), self.homography)[0][0]

            rw_goalPoint = np.array([[self.GOAL_CENTER,0]],dtype=np.float32).reshape(-1,1,2)
            img_goalPoint = cv.perspectiveTransform(np.array(rw_goalPoint), np.linalg.inv(self.homography))[0][0]
            
            distToGoal = np.linalg.norm(rw_ballPoint-rw_goalPoint)

            img_size = (len(self.img[0]),len(self.img))

            blank_image = np.zeros((len(self.img[0]) * self.QUALITY_MULTIPLIER,len(self.img) * self.QUALITY_MULTIPLIER,4), np.uint8)
            tuple_ballPoint = tuple(np.array(rw_ballPoint, dtype=np.int))
            tuple_goalPoint = tuple(np.array(rw_goalPoint[0][0], dtype=np.int))
            arrow = cv.arrowedLine(blank_image,tuple_ballPoint,tuple_goalPoint,(255,0,0,255),thickness=10)

            transformedArrow = cv.warpPerspective(arrow, np.linalg.inv(self.homography), img_size)

            b_channel, g_channel, r_channel = cv.split(self.img)
            alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255
            img_rgba = cv.merge((b_channel, g_channel, r_channel, alpha_channel))

            self.img = cv.addWeighted(img_rgba,1,transformedArrow,1,0)

            cv.imshow('Arrow', transformedArrow)
            cv.imshow('Select 4 Points', self.img) 

            print(distToGoal)
            
class GoalCircle(Homo):
    def click_event(self, event, x, y, flags, param):
        if (super().click_event(event,x,y,flags,param)):
            rw_goalPoint = np.array([[self.GOAL_CENTER,0]],dtype=np.float32).reshape(-1,1,2)

            img_goalPoint = cv.perspectiveTransform(np.array(rw_goalPoint), np.linalg.inv(self.homography))[0][0]
            tuple_goalPoint = tuple([int(img_goalPoint[0]),int(img_goalPoint[1])])

            cv.circle(self.img,tuple_goalPoint,10,(0,255,0))  
            cv.imshow('Select 4 Points', self.img) 

def loadImage(filename):
    img = cv.imread(filename)
    
    if debug:
        cv.imshow('Select 4 Points', img)

    return img

if __name__ == "__main__":
    for i in range(1, 4):
        filename = 'images/{}.png'.format(i)
        img = loadImage(filename)

        homo = GoalArrowDistance(img)
        #homo = GoalCircle(img)

        cv.setMouseCallback('Select 4 Points', homo.click_event)

        cv.waitKey(0)
    cv.destroyAllWindows()