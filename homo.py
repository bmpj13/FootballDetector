import cv2 as cv
import numpy as np
import random
import math
from canny import compute

debug = True

#Offsets allow for drawings to take up the whole field
X_OFFSET = 50
Y_OFFSET = 10

RW_DATABASE = [
    [0 + X_OFFSET,0 + Y_OFFSET], #1
    [0 + X_OFFSET,5.5 + Y_OFFSET], #2
    [0 + X_OFFSET,16.5 + Y_OFFSET], #3
    [11 + X_OFFSET,0 + Y_OFFSET], #4
    [11 + X_OFFSET,5.5 + Y_OFFSET], #5
    [11 + X_OFFSET,16.5 + Y_OFFSET], #6
    [29.32 + X_OFFSET,0 + Y_OFFSET], #7
    [29.32 + X_OFFSET,5.5 + Y_OFFSET], #8
    [29.32 + X_OFFSET,16.5 + Y_OFFSET], #9
    [40.32 + X_OFFSET,0 + Y_OFFSET], #10
    [40.32 + X_OFFSET,5.5 + Y_OFFSET], #11
    [40.32 + X_OFFSET,16.5 + Y_OFFSET] #12
]

class Homo:
    picked = None
    dragging = False
    window_name = 'Select Interest Points'

    homography_points = []
    homography = []
    img = []

    QUALITY_MULTIPLIER = 40
    GOAL_CENTER = [(20.16 + X_OFFSET) * QUALITY_MULTIPLIER, (Y_OFFSET) * QUALITY_MULTIPLIER]

    def __init__(self, img, homography_points):
        self.img = img
        self.homography_points = points

    def handleMouse(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            self.picked = None
            for i, point in enumerate(self.homography_points):
                if self.should_pick(point, (x,y)):
                    self.picked = (i, point[0], point[1])
                    self.dragging = True

            if self.picked is None:
                new_point = (None, (x, y))
                self.homography_points.append(new_point)
                self.picked = (len(self.homography_points)-1, None, (x, y))

        elif event == cv.EVENT_MOUSEMOVE:
            if self.dragging:
                (i, point_id, _) = self.picked
                self.picked = (i, point_id, (x, y))
                self.homography_points[i] = (point_id, (x,y))

        elif event == cv.EVENT_LBUTTONUP:
            self.dragging = False

    def handleKeyboard(self, key):
        if key == ord('1'):
            self.update_picked_id(1)
        elif key == ord('2'):
            self.update_picked_id(2)
        elif key == ord('3'):
            self.update_picked_id(3)
        elif key == ord('4'):
            self.update_picked_id(4)
        elif key == ord('5'):
            self.update_picked_id(5)
        elif key == ord('6'):
            self.update_picked_id(6)
        elif key == ord('7'):
            self.update_picked_id(7)
        elif key == ord('8'):
            self.update_picked_id(8)
        elif key == ord('9'):
            self.update_picked_id(9)
        elif key == ord('q'):
            self.update_picked_id(10)
        elif key == ord('w'):
            self.update_picked_id(11)
        elif key == ord('e'):
            self.update_picked_id(12)
        elif key == 8:
            # backspace
            self.remove_picked_point()
        elif key == 13:
            return True

        return False

    def should_pick(self, point, input_coords):
        x, y = input_coords
        _, (px, py) = point

        return abs(x-px) < 7 and abs(y-py) < 7

    def is_picked(self, point):
        if self.picked is not None:
            _, (x, y) = point
            x, y = int(x), int(y)
            (_, _, (px, py)) = self.picked

            if abs(x-px) < 7 and abs(y-py) < 7:
                return True
        return False

    def update_picked_id(self, new_id):
        for i, point in enumerate(self.homography_points):
            point_id, (x, y) = point
            if point_id == new_id and not self.is_picked(point):
                self.homography_points[i] = (None, (x, y))

            if self.is_picked(point):
                self.picked = (i, new_id, (x, y))
                self.homography_points[i] = (new_id, (x, y))
    
    def remove_picked_point(self):
        if self.picked is not None:
            (index, point_id, (x, y)) = self.picked

            self.homography_points.pop(index)
            self.picked = None
    
    def show(self):
        cv.imshow(self.window_name, self.img)
        cv.setMouseCallback(self.window_name, self.handleMouse)
        self.show_picker_loop()

    def show_picker_loop(self):
        while True:
            k = cv.waitKey(1)

            stop = self.handleKeyboard(k)
            if stop:
                break

            img = self.img.copy()
            for point in self.homography_points:
                point_id, (x, y) = point
                x, y = int(x), int(y)
                color = (0, 0, 255) if self.is_picked(point) else (255, 0, 0)

                id_text = str(point_id)
                num_chars = len(id_text)

                cv.circle(img, (x, y), 7, color, 1)
                cv.circle(img, (x, y), 2, color, -1)
                cv.putText(img, id_text, (x - 10 - 10 * num_chars, y + 5), cv.FONT_HERSHEY_PLAIN, 1, color)

            cv.imshow(self.window_name, img)
        
        self.action()
    
    def action(self):
        self.build_homography()
        print("Homography matrix built.")

    def build_homography(self):
        img_points = []
        rw_points = []

        for (point_id,(x,y)) in self.homography_points:
            if point_id == None:
                continue
            img_points.append([x,y])
            rw_points.append(RW_DATABASE[point_id-1])

        rw_points_scaled = np.array(rw_points,dtype=np.float32) * self.QUALITY_MULTIPLIER
        self.homography, status = cv.findHomography(np.array(img_points), rw_points_scaled)

class GoalCircle(Homo):
    def action(self):
        super().action()
        rw_goalPoint = np.array([self.GOAL_CENTER],dtype=np.float32).reshape(-1,1,2)

        img_goalPoint = cv.perspectiveTransform(np.array(rw_goalPoint), np.linalg.inv(self.homography))[0][0]
        tuple_goalPoint = tuple([int(img_goalPoint[0]),int(img_goalPoint[1])])

        cv.circle(self.img,tuple_goalPoint,10,(0,255,0))  
        cv.imshow('Goal Circle', self.img) 

class GoalArrowDistance(Homo):
    def handleBallPlacement(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            img_ballPoint = np.array([[float(x),float(y)]]).reshape(-1,1,2)
            rw_ballPoint = cv.perspectiveTransform(np.array(img_ballPoint), self.homography)[0][0]

            rw_goalPoint = np.array([self.GOAL_CENTER],dtype=np.float32).reshape(-1,1,2)
            img_goalPoint = cv.perspectiveTransform(np.array(rw_goalPoint), np.linalg.inv(self.homography))[0][0]
            
            distToGoal = round(np.linalg.norm(rw_ballPoint-rw_goalPoint)/self.QUALITY_MULTIPLIER,2)

            img_size = (len(self.img[0]),len(self.img))

            blank_image = np.zeros((len(self.img[0]) * self.QUALITY_MULTIPLIER,len(self.img) * self.QUALITY_MULTIPLIER), np.uint8)
            tuple_ballPoint = tuple(np.array(rw_ballPoint, dtype=np.int))
            tuple_goalPoint = tuple(np.array(rw_goalPoint[0][0], dtype=np.int))
            arrow = cv.arrowedLine(blank_image,tuple_ballPoint,tuple_goalPoint,(255),thickness=20)

            transformedArrow = cv.warpPerspective(arrow, np.linalg.inv(self.homography), img_size)
            transformedArrow = cv.GaussianBlur(transformedArrow,(7,7),sigmaX=0,sigmaY=0)

            b_channel, g_channel, r_channel = cv.split(self.img)
            alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255
            img_rgba = cv.merge((b_channel, g_channel, r_channel, alpha_channel))

            transformedArrowBGR = cv.cvtColor(transformedArrow, cv.COLOR_GRAY2BGRA)
            transformedArrowBGR[:, :, 1] = 0
            transformedArrowBGR[:, :, 2] = 0
            transformedArrowBGR[:, :, 3] = transformedArrowBGR[:, :, 0] #componente de azul

            self.img = cv.addWeighted(img_rgba,1,transformedArrowBGR,1,0)
            
            distText = f"{distToGoal}m"

            cv.putText(self.img, distText, (x - 10 - 10 * len(distText), y + 5), cv.FONT_HERSHEY_PLAIN, 1, (255,255,255))

            cv.imshow('Arrow', transformedArrow)
            cv.imshow('Goal Arrow Distance', self.img) 

    def action(self):
        super().action()
        cv.imshow('Goal Arrow Distance', self.img) 
        cv.setMouseCallback('Goal Arrow Distance', self.handleBallPlacement)

class OffsideLine(Homo):
    def handlePointPlacement(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            img_offsidePoint = np.array([[float(x),float(y)]]).reshape(-1,1,2)
            rw_offsidePoint_y = cv.perspectiveTransform(np.array(img_offsidePoint), self.homography)[0][0][1] #Only the y component matters

            img_size = (len(self.img[0]),len(self.img))

            blank_image = np.zeros((len(self.img[0]) * self.QUALITY_MULTIPLIER,len(self.img) * self.QUALITY_MULTIPLIER,4), np.uint8)
            tuple_startPoint = tuple(np.array([0,rw_offsidePoint_y], dtype=np.int))
            tuple_endPoint = tuple(np.array([len(self.img[0]) * self.QUALITY_MULTIPLIER,rw_offsidePoint_y], dtype=np.int))
            line = cv.line(blank_image,tuple_startPoint,tuple_endPoint,(255,0,0,255),thickness=10)

            transformedLine= cv.warpPerspective(line, np.linalg.inv(self.homography), img_size)

            b_channel, g_channel, r_channel = cv.split(self.img)
            alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255
            img_rgba = cv.merge((b_channel, g_channel, r_channel, alpha_channel))

            self.img = cv.addWeighted(img_rgba, 1, transformedLine, 1, 0)
            
            #cv.imshow('Line', transformedLine)
            cv.imshow('OffsideLine', self.img) 

    def action(self):
        super().action()
        cv.imshow('Offside Line', self.img) 
        cv.setMouseCallback('Offside Line', self.handlePointPlacement)

if __name__ == "__main__":
    for i in range(1, 4):
        filename = 'images/{}.png'.format(i)
        window_name = 'Select Interest Points'

        img, _, _, points = compute(filename, use_debug=False)
        homo = GoalArrowDistance(img, points)
        homo.show()
    cv.destroyAllWindows()