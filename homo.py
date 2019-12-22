import cv2 as cv
import numpy as np
import random
import math
import argparse
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
    window_name = ''

    homography_points = []
    homography = []
    img = []
    player_mask = []
    field_mask = []

    QUALITY_MULTIPLIER = 40
    GOAL_CENTER = [(20.16 + X_OFFSET) * QUALITY_MULTIPLIER, (0.25 + Y_OFFSET) * QUALITY_MULTIPLIER]

    def __init__(self, window_name, img, homography_points, players, field):
        self.window_name = window_name
        self.img = img
        self.homography_points = points
        self.player_mask = players
        self.field_mask = field

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
            # return
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

class GoalArrowDistance(Homo):
    def handleBallPlacement(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            # Ball coords
            img_ballPoint = np.array([[ float(x), float(y) ]]).reshape(-1,1,2)
            rw_ballPoint = cv.perspectiveTransform(np.array(img_ballPoint), self.homography)[0][0]

            # Goal coords
            rw_goalPoint = np.array([ self.GOAL_CENTER ], dtype=np.float32).reshape(-1,1,2)
            img_goalPoint = cv.perspectiveTransform(np.array(rw_goalPoint), np.linalg.inv(self.homography))[0][0]
            
            # Distance
            distToGoal = round(np.linalg.norm(rw_ballPoint - rw_goalPoint) / self.QUALITY_MULTIPLIER, 2) + 0.25

            # Draw arrow on real world and convert back to the image space
            img_size = (self.img.shape[1], self.img.shape[0])
            blank_image = np.zeros((len(self.img[0]) * self.QUALITY_MULTIPLIER, len(self.img) * self.QUALITY_MULTIPLIER), np.uint8)

            tuple_ballPoint = tuple(np.array(rw_ballPoint, dtype=np.int))
            tuple_goalPoint = tuple(np.array(rw_goalPoint[0][0], dtype=np.int))
            arrow = cv.arrowedLine(blank_image, tuple_ballPoint, tuple_goalPoint, 255, thickness=20)

            transformedArrow = cv.warpPerspective(arrow, np.linalg.inv(self.homography), img_size)
            transformedArrow = cv.GaussianBlur(transformedArrow, (5,5), sigmaX=0, sigmaY=0)

            transformedArrowBGR = cv.cvtColor(transformedArrow, cv.COLOR_GRAY2BGR)
            transformedArrowBGR[:, :, 0] = transformedArrow
            transformedArrowBGR[:, :, 1] = 0
            transformedArrowBGR[:, :, 2] = 0
            
            players = self.player_mask
            players_inv = cv.bitwise_not(players)

            image = self.img.copy()
            image = cv.bitwise_and(image, image, mask=players_inv)
            image = cv.addWeighted(image, 1, transformedArrowBGR, 1, 0)
            image[players > 0] = self.img[players > 0]


            distText = f"{distToGoal}m"
            cv.putText(image, distText, (x - 10 - 10 * len(distText), y + 5), cv.FONT_HERSHEY_PLAIN, 1, (255,255,255))
            cv.imshow(self.window_name, image) 

    def action(self):
        super().action()
        cv.imshow(self.window_name, self.img) 
        cv.setMouseCallback(self.window_name, self.handleBallPlacement)

class OffsideLine(Homo):
    def handlePointPlacement(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            img_offsidePoint = np.array([[float(x),float(y)]]).reshape(-1,1,2)
            rw_offsidePoint_y = cv.perspectiveTransform(np.array(img_offsidePoint), self.homography)[0][0][1] #Only the y component matters

            img_size = (len(self.img[0]),len(self.img))

            blank_image = np.zeros((len(self.img[0]) * self.QUALITY_MULTIPLIER,len(self.img) * self.QUALITY_MULTIPLIER), np.uint8)
            tuple_startPoint = tuple(np.array([0,rw_offsidePoint_y], dtype=np.int))
            tuple_endPoint = tuple(np.array([len(self.img[0]) * self.QUALITY_MULTIPLIER,rw_offsidePoint_y], dtype=np.int))
            line = cv.line(blank_image,tuple_startPoint,tuple_endPoint, 255,thickness=10)

            transformedLine = cv.warpPerspective(line, np.linalg.inv(self.homography), img_size)

            transformedLineBGR = cv.cvtColor(transformedLine, cv.COLOR_GRAY2BGR)
            transformedLineBGR[:, :, 0] = transformedLine
            transformedLineBGR[:, :, 1] = 0
            transformedLineBGR[:, :, 2] = 0

            blank_image = np.zeros((len(self.img[0]) * self.QUALITY_MULTIPLIER,len(self.img) * self.QUALITY_MULTIPLIER), np.uint8)
            rectangle = cv.rectangle(blank_image, tuple_endPoint, tuple(np.array([0, Y_OFFSET * self.QUALITY_MULTIPLIER], dtype=np.int)), 255, -1)
            transformedRectangle = cv.warpPerspective(rectangle, np.linalg.inv(self.homography), img_size)
            transformedRectangleBGR = cv.cvtColor(transformedRectangle, cv.COLOR_GRAY2BGR)
            
            players = self.player_mask
            players_inv = cv.bitwise_not(players)
            field = self.field_mask

            image = self.img.copy()
            image = cv.bitwise_and(image, image, mask=players_inv)
            image[field > 0] = cv.addWeighted(image[field > 0], 1, transformedLineBGR[field > 0], 1, 0)
            image[field > 0] = cv.addWeighted(image[field > 0], 1, transformedRectangleBGR[field > 0], -0.1, 0)
            image[players > 0] = self.img[players > 0]

            cv.imshow(self.window_name, image) 

    def action(self):
        super().action()
        cv.imshow(self.window_name, self.img) 
        cv.setMouseCallback(self.window_name, self.handlePointPlacement)


class FreeKickCircle(Homo):
    def handlePointPlacement(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            img_ballPoint = np.array([[float(x),float(y)]]).reshape(-1,1,2)
            rw_ballPoint = cv.perspectiveTransform(np.array(img_ballPoint), self.homography)[0][0]

            img_size = (len(self.img[0]),len(self.img))

            tuple_ballPoint = tuple(np.array(rw_ballPoint, dtype=np.int))
            blank_image = np.zeros((len(self.img[0]) * self.QUALITY_MULTIPLIER,len(self.img) * self.QUALITY_MULTIPLIER), np.uint8)
            circumference = cv.circle(blank_image, tuple_ballPoint, 9 * self.QUALITY_MULTIPLIER, 255, 10)

            transformedCircumference = cv.warpPerspective(circumference, np.linalg.inv(self.homography), img_size)
            transformedCircumference = cv.GaussianBlur(transformedCircumference, (5,5), sigmaX=0, sigmaY=0)
            transformedCircumferenceBGR = cv.cvtColor(transformedCircumference, cv.COLOR_GRAY2BGR)

            blank_image = np.zeros((len(self.img[0]) * self.QUALITY_MULTIPLIER,len(self.img) * self.QUALITY_MULTIPLIER), np.uint8)
            circle = cv.circle(blank_image, tuple_ballPoint, 9 * self.QUALITY_MULTIPLIER, 255, -1)
            transformedCircle = cv.warpPerspective(circle, np.linalg.inv(self.homography), img_size)
            transformedCircleBGR = cv.cvtColor(transformedCircle, cv.COLOR_GRAY2BGR)
            
            players = self.player_mask
            players_inv = cv.bitwise_not(players)
            field = self.field_mask

            image = self.img.copy()
            image = cv.bitwise_and(image, image, mask=players_inv)
            image[field > 0] = cv.addWeighted(image[field > 0], 1, transformedCircumferenceBGR[field > 0], 1, 0)
            image[field > 0] = cv.addWeighted(image[field > 0], 1, transformedCircleBGR[field > 0], -0.1, 0)
            image[players > 0] = self.img[players > 0]
            
            cv.imshow(self.window_name, image) 

    def action(self):
        super().action()
        cv.imshow(self.window_name, self.img) 
        cv.setMouseCallback(self.window_name, self.handlePointPlacement)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drawing stuff on a football pitch")
    parser.add_argument('--image', help="Path to image", type=str, required=True)
    parser.add_argument('--type', help="Type of AR functionality", choices=('arrow', 'offside', 'circle'), type=str, required=True)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    filename = args.image
    img, points, players, field = compute(filename, use_debug=args.debug)

    if args.type == 'arrow':
        homo = GoalArrowDistance('Goal Arrow Distance', img, points, players, field)
    elif args.type == 'offside':
        homo = OffsideLine('Offside Line', img, points, players, field)
    elif args.type == 'circle':
        homo = FreeKickCircle('Free Kick Circle', img, points, players, field)
    else:
        print("Type not available")
        exit(1)

    homo.show()
    cv.waitKey(0)