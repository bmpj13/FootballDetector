import cv2 as cv
import numpy as np
import random

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

def fitFieldLines(img_contours, toleranceM = 0.1, toleranceB = 20):
    field_lines = []
    for img_contour in img_contours:
        lines = cv.HoughLinesP(img_contour, 1, 1*np.pi/180, 80, minLineLength=10, maxLineGap=20)
        lines = lines[:, 0, :] if lines is not None else []

        # Group together similar lines (with similar 'm' and 'b' in eq. y = m*x + b) in bins
        bins = []
        for x1, y1, x2, y2 in lines:
            if x2-x1 == 0:
                continue
            m = (y2-y1)/(x2-x1)
            b = y1 - m*x1
            i = 0
            while i < len(bins):
                bin_m, bin_b, bin_lines = bins[i]
                if abs(bin_m-m) < toleranceM and abs(bin_b-b) < toleranceB:
                    bin_lines.append((x1, y1, x2, y2))
                    bins[i] = ( (bin_m+m)/2, (bin_b+b)/2, bin_lines )
                    break
                i += 1
            if i == len(bins):
                bins.append((m, b, [(x1, y1, x2, y2)]))

        # Rearrange data to use in fitLine function. 
        # field_points is a list of lists (each entry corresponds to the points found in each bin)
        field_points = []
        for _, _, lines in bins:
            points = []
            for line in lines:
                x1, y1, x2, y2 = line
                points.append((x1, y1))
                points.append((x2, y2))
            points = np.array(points)
            field_points.append(points)
        field_points = np.array(field_points)

        # Compute, for each bin, the best line match
        lines = []
        for points in field_points:
            regression = cv.fitLine(points, cv.DIST_L2, 0, 0.01, 0.01)[:,0]
            lines.append(regression)
        lines = np.array(lines)
        field_lines.append(lines)
    field_lines = np.array(field_lines)

    if debug:
        img_copy = np.zeros(img_contour.shape, dtype=np.uint8)
        img_copy = cv.cvtColor(img_copy, cv.COLOR_GRAY2BGR)
        for img_contour in img_contours:
            img_copy = cv.bitwise_or(img_copy, cv.cvtColor(img_contour, cv.COLOR_GRAY2BGR))
            for lines in field_lines:
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                for line in lines:
                    vx, vy, x0, y0 = line
                    x1, y1 = int(x0 - 1000*vx), int(y0 - 1000*vy)
                    x2, y2 = int(x0 + 1000*vx), int(y0 + 1000*vy)
                    cv.line(img_copy, (x1,y1), (x2,y2), color, 1)
        cv.imshow('Fitted Field Lines', img_copy)

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
    field_lines = fitFieldLines(img_contours)

    return img_contours, field_lines


def vector2cartesian(line):
    vx, vy, x0, y0 = line
    x1, y1 = x0 + 5*vx, y0 + 5*vy
    slope = (y1-y0) / (x1-x0)
    axisy = y0 - slope * x0
    return slope, axisy

def getInterestPoints(img_contours, field_lines):
    intersection_points = []
    for img_contour, contour_field_lines in zip(img_contours, field_lines):
        height, width = img_contour.shape
        for i in range(len(contour_field_lines)):
            line1 = contour_field_lines[i]
            a, c = vector2cartesian(line1) # y = a*x + c
            
            for j in range(i+1, len(contour_field_lines)):
                line2 = contour_field_lines[j]
                b, d = vector2cartesian(line2) # y = b*x + d

                if a == b: # lines are parallel
                    continue
                
                x = (d-c) / (a-b)
                y = (a*d - b*c) / (a-b)

                if (x >= 0 and x <= width) and (y >= 0 and y <= height):
                    intersection_points.append((x,y))
    intersection_points = np.array(intersection_points)

    if debug:
        img_copy = np.zeros(img_contour.shape, dtype=np.uint8)
        img_copy = cv.cvtColor(img_copy, cv.COLOR_GRAY2BGR)
        for img_contour in img_contours:
            img_copy = cv.bitwise_or(img_copy, cv.cvtColor(img_contour, cv.COLOR_GRAY2BGR))
        for x,y in intersection_points:
            cv.circle(img_copy, (int(x), int(y)), 3, (255, 0, 0), -1)
        cv.imshow('Interest Points', img_copy)

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
    img_contours, field_lines = getField(gradient)
    points = getInterestPoints(img_contours, field_lines)

    # IN PROGRESS:
    H, H_inv = getHomography(points)
    test = cv.perspectiveTransform(np.float32([0,11]).reshape(1, 1, -1), H_inv)
    x,y = test[0,0,:]
    print(x,y)
    cv.circle(img, (x,y), 3, (255, 0, 0), -1)
    cv.imshow('TEST', img)
    cv.waitKey(0)


cv.destroyAllWindows()