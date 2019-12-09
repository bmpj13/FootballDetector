import cv2 as cv
import numpy as np
import random
import math
from canny import compute

debug = True

class Homo:
    picked = None
    dragging = False
    window_name = 'Select Interest Points'

    homography_points = []
    homography = []
    img = []

    QUALITY_MULTIPLIER = 30
    GOAL_CENTER = 20.16 * QUALITY_MULTIPLIER

    def __init__(self, img, homography_points):
        self.img = img
        self.homography_points = points

    def calculate_homography(self):
        rw_points = np.array([[11,5.5],[29.32,5.5],[12.85,16.5],[27.47,16.5]],dtype=np.float32) * self.QUALITY_MULTIPLIER
        self.homography, status = cv.findHomography(np.array(self.homography_points), rw_points)

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
        if key == ord('0'):
            self.update_picked_id(0)
        elif key == ord('1'):
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
    
    def build_homography(self):
        #TODO - Build the homography matrix here
        pass

    def action(self):
        self.build_homography()
        print("Homography matrix built.")

class GoalCircle(Homo):
    def action(self):
        super().action()
        rw_goalPoint = np.array([[self.GOAL_CENTER,0]],dtype=np.float32).reshape(-1,1,2)

        img_goalPoint = cv.perspectiveTransform(np.array(rw_goalPoint), np.linalg.inv(self.homography))[0][0]
        tuple_goalPoint = tuple([int(img_goalPoint[0]),int(img_goalPoint[1])])

        cv.circle(self.img,tuple_goalPoint,10,(0,255,0))  
        cv.imshow('Goal Circle', self.img) 

class GoalArrowDistance(Homo):
    def action(self):
        super().action()
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
        cv.imshow('Goal Arrow Distance', self.img) 

if __name__ == "__main__":
    for i in range(1, 4):
        filename = 'images/{}.png'.format(i)
        window_name = 'Select Interest Points'

        img, _, _, points = compute(filename, use_debug=False)
        homo = GoalArrowDistance(img, points)
        homo.show()
    cv.destroyAllWindows()