import cv2 as cv
import numpy as np
import random
import math

debug = True

def loadImage(filename):
    img = cv.imread(filename)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    if debug:
        cv.imshow('Image', img)
        cv.imshow('Gray Image', img_gray)

    return img, img_gray

def imageGradient(img):
    blur = cv.blur(img, ksize=(3,3))
    sobelx = cv.Sobel(blur, cv.CV_64F, 1, 0, ksize=-1)
    sobely = cv.Sobel(blur, cv.CV_64F, 0, 1, ksize=-1)
    gradient = cv.add(sobelx, sobely)
    gradient = cv.convertScaleAbs(gradient)

    if debug:
        cv.imshow('Blur', blur)
        cv.imshow('Sobel X', cv.convertScaleAbs(sobelx))
        cv.imshow('Sobel Y', cv.convertScaleAbs(sobely))
        cv.imshow('Gradient', gradient)
    
    return gradient

def findFieldContours(img):
    contours, _ = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv.contourArea, reverse=True)[0:3]
    return contours

def getFieldLines(img, contours):
    approx_points = []
    img_contours = []
    for contour in contours:
        img_contour = np.zeros(img.shape, dtype=np.uint8)
        approx = cv.approxPolyDP(contour, 0.005 * cv.arcLength(contour, True), True)

        cv.drawContours(img_contour, [approx], -1, 255, 1)
        img_contour = cv.GaussianBlur(img_contour, (3, 3), sigmaX=0, sigmaY=0)

        approx_points.append(approx)
        img_contours.append(img_contour)

    if debug:
        contours_img1 = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        contours_img2 = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        cv.drawContours(contours_img1, contours, -1, (0, 0, 255), 2)
        cv.drawContours(contours_img2, approx_points, -1, (0, 0, 255), 2)
        cv.imshow('Contours', contours_img1)
        cv.imshow('Approx. Contours', contours_img2)
    
    return approx_points, img_contours

def groupLines(img_contours, toleranceM = 0.2, toleranceB = 30, tolerancePixels = 10):
    bins = []

    # Group together similar lines (with similar 'm' and 'b' in eq. y = m*x + b) in bins
    for img_contour in img_contours:
        # Find lines in image using HoughLinesP
        lines = cv.HoughLinesP(img_contour, 1, 1 * np.pi/180, 80, minLineLength=10, maxLineGap=20)
        lines = lines[:, 0, :] if lines is not None else []
        lines = filter(lambda x: x[0] - x[2] != 0, lines) # filter vertical lines

        for x1, y1, x2, y2 in lines:
            m = (y2-y1)/(x2-x1)
            b = y1 - m*x1
            i = 0
            while i < len(bins):
                bin_m, bin_b, bin_min_x, bin_min_y, bin_max_x, bin_max_y, bin_lines = bins[i]
                if abs(bin_m-m) < toleranceM and abs(bin_b-b) < toleranceB:
                    bin_lines.append((x1, y1, x2, y2))
                    mean_slope = (bin_m+m)/2
                    mean_axisy = (bin_b+b)/2
                    min_x = min(bin_min_x, x1, x2)
                    min_y = min(bin_min_y, y1, y2)
                    max_x = max(bin_max_x, x1, x2)
                    max_y = max(bin_max_y, y1, y2)
                    bins[i] = (mean_slope, mean_axisy, min_x, min_y, max_x, max_y, bin_lines)
                    break
                i += 1
            if i == len(bins):
                bins.append((m, b, min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2), [(x1, y1, x2, y2)]))
    bins = np.array(bins)
    
    if debug:
        img_copy = cv.cvtColor(img_contours[0], cv.COLOR_GRAY2BGR)
        for img_contour in img_contours[1:]:
            img_copy = cv.bitwise_or(img_copy, cv.cvtColor(img_contour, cv.COLOR_GRAY2BGR))
        for _, _, min_x, min_y, max_x, max_y, lines in bins:
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            for line in lines:
                x1, y1, x2, y2 = line
                cv.line(img_copy, (x1, y1), (x2, y2), color, 2)
                #cv.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
        cv.imshow('Grouped Lines', img_copy)

    return bins
    

def fitFieldLines(img_contours, bins):
    # Rearrange data to use in fitLine function. 
    # field_points is a list of lists (each entry corresponds to the points found in each bin)
    field_points = []
    for _, _, _, _, _, _, lines in bins:
        points = []
        for line in lines:
            x1, y1, x2, y2 = line
            points.append((x1, y1))
            points.append((x2, y2))
        field_points.append(np.array(points))
    field_points = np.array(field_points)

    # Compute, for each bin, the best line match
    field_lines = []
    for points in field_points:
        regression = cv.fitLine(points, cv.DIST_L2, 0, 0.01, 0.01)[:,0]
        field_lines.append(np.array(regression))
    field_lines = np.array(field_lines)

    if debug:
        img_copy = np.zeros(img_contours[0].shape, dtype=np.uint8)
        img_copy = cv.cvtColor(img_copy, cv.COLOR_GRAY2BGR)
        for img_contour in img_contours:
            img_copy = cv.bitwise_or(img_copy, cv.cvtColor(img_contour, cv.COLOR_GRAY2BGR))
        for line in field_lines:
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            vx, vy, x0, y0 = line
            x1, y1 = int(x0 - 1000*vx), int(y0 - 1000*vy)
            x2, y2 = int(x0 + 1000*vx), int(y0 + 1000*vy)
            cv.line(img_copy, (x1,y1), (x2,y2), color, 1)
        cv.imshow('Fitted Field Lines', img_copy)
        cv.waitKey(0)

    return field_lines

def getField(gradient):
    # We expect the field to have a small gradient
    _, gradient_threshold = cv.threshold(gradient, 80, 255, cv.THRESH_BINARY_INV)

    # Remove noise and improve object separation
    kernel = np.ones((9,9), np.uint8)
    opening = cv.morphologyEx(gradient_threshold, cv.MORPH_OPEN, kernel)

    if debug:
        cv.imshow('Gradient Threshold', gradient_threshold)
        cv.imshow('Opening', opening)

    contours = findFieldContours(opening)
    _, img_contours = getFieldLines(opening, contours)
    bins = groupLines(img_contours)
    field_lines = fitFieldLines(img_contours, bins)

    return img_contours, field_lines, bins

def vector2cartesian(line):
    vx, vy, x0, y0 = line
    x1, y1 = x0 + 5*vx, y0 + 5*vy
    slope = (y1-y0) / (x1-x0)
    axisy = y0 - slope * x0
    return slope, axisy

def getInterestPoints(img_contours, field_lines, bins):
    assert len(field_lines) == len(bins)

    intersection_points = []
    height, width = img_contours[0].shape
    for i in range(len(field_lines)):
        for j in range(i + 1, len(field_lines)):
            l1 = field_lines[i]
            b1 = bins[i]
            l2 = field_lines[j]
            b2 = bins[j]

            a, c = vector2cartesian(l1)
            b, d = vector2cartesian(l2)
            if a == b: # lines are parallel
                continue

            x = (d-c) / (a-b)
            y = (a*d - b*c) / (a-b)

            _, _, min_x1, min_y1, max_x1, max_y1, _ = b1
            _, _, min_x2, min_y2, max_x2, max_y2, _ = b2

            if not (x >= 0 and x <= width and y >= 0 and y <= height): # intersection not in image
                continue

            if min_x1 > max_x2 or min_y1 > max_y2 or min_x2 > max_x1 or min_y2 > max_y1: # rectangles don't meet (see Interest Points window)
                continue

            min_x = max(min_x1, min_x2) - 25
            min_y = max(min_y1, min_y2) - 25
            max_x = min(max_x1, max_x2) + 25
            max_y = min(max_y1, max_y2) + 25
            
            if not (min_x <= x <= max_x and min_y <= y <= max_y):   # interest points far away from rectangle intersection (see Interest Points window)
                continue

            intersection_points.append((x,y))

            if debug:
                img_copy = np.zeros(img_contours[0].shape, dtype=np.uint8)
                img_copy = cv.cvtColor(img_copy, cv.COLOR_GRAY2BGR)
                for img_contour in img_contours:
                    img_copy = cv.bitwise_or(img_copy, cv.cvtColor(img_contour, cv.COLOR_GRAY2BGR))
                for x,y in intersection_points:
                    cv.circle(img_copy, (int(x), int(y)), 6, (255, 0, 0), -1)
                cv.rectangle(img_copy, (b1[2], b1[3]), (b1[4], b1[5]), (0, 255, 0), 2)
                cv.rectangle(img_copy, (b2[2], b2[3]), (b2[4], b2[5]), (0, 255, 0), 2)
                cv.rectangle(img_copy, (min_x, min_y), (max_x, max_y), (0, 255, 255), 2)
                cv.imshow('Interest Points', img_copy)
                cv.waitKey(0)
    intersection_points = np.array(intersection_points)

    return intersection_points

def getHomography(points):
    real_world_coords = np.array([
        (0, 0),
        (0, 11),
        (0, 16.5),
        (0, 23.82),
        (0, 29.32),
        (0, 40.32),
        (16.5, 0),
        (5.5, 11),
        (5.5, 29.32),
        (16.5, 40.32)
    ])

    pts_src = np.float32(points).reshape(len(points), 1, -1)
    pts_dst = np.float32(real_world_coords).reshape(len(real_world_coords), 1, -1)

    pts_src = pts_src[0:len(pts_dst)]

    H = cv.findHomography(pts_src, pts_dst, cv.RANSAC)[0]
    H_inv = np.linalg.inv(H)

    T_src = cv.perspectiveTransform(pts_src, H)
    print(pts_dst)
    print(T_src)

    return H, H_inv
    


for i in range(1, 4):
    filename = 'images/{}.png'.format(i)

    img, img_gray = loadImage(filename)
    gradient = imageGradient(img_gray)
    img_contours, field_lines, bins = getField(gradient)
    points = getInterestPoints(img_contours, field_lines, bins)

    for i in range(len(points)):
        for j in range(i+1,len(points)):
            p1 = points[i]
            p2 = points[j]
            dx = p1[0] - p2[0]
            dy = p1[1] - p2[1]
            distance = math.sqrt(dx*dx + dy*dy)
            print(distance)

    cv.waitKey(0)


cv.destroyAllWindows()