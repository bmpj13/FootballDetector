import cv2 as cv
import numpy as np
import random
import math

debug = True

def get_box_values(box):
    (x, y), (width, height), angle = box
    x, y, width, height = x, y, width, height

    if angle < -45.0:     # Switch height and width if angle between -90 and -45 ( width, height 90º = height, width 0º )
        width, height = height, width

    return x, y, width, height, angle

def should_group(group, line, toleranceM, minPointDistance):
    should_group = False

    (avg_vx, avg_vy), group_lines = group
    avg_m = avg_vy / avg_vx
    
    vx, vy, x0, y0 = line
    m = vy / vx

    if abs(avg_m-m) < toleranceM:
        j = 0
        while j < len(group_lines):
            _, _, tx0, ty0 = group_lines[j]
            dx = tx0 - x0
            dy = ty0 - y0
            if math.sqrt(dx*dx + dy*dy) < minPointDistance:
                break
            j += 1
        if j == len(group_lines):
            should_group = True
    
    return should_group

def cross_product(vec1, vec2):
    (x1, y1) = vec1
    (x2, y2) = vec2
    return (x1 * y2) - (x2 * y1)

def vector2cartesian(line):
    vx, vy, x0, y0 = line
    x1, y1 = x0 + 5*vx, y0 + 5*vy
    slope = (y1-y0) / (x1-x0)
    axisy = y0 - slope * x0
    return slope, axisy



def loadImage(filename):
    img = cv.imread(filename)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    if debug:
        cv.imshow('Image', img)
        cv.imshow('Gray Image', img_gray)

    return img, img_gray


def getField(img_gray):
    # Compute image's gradient
    blur = cv.blur(img_gray, ksize=(5,5))
    sobelx = cv.Sobel(blur, cv.CV_64F, 1, 0, ksize=-1)
    sobely = cv.Sobel(blur, cv.CV_64F, 0, 1, ksize=-1)
    gradient = cv.add(sobelx, sobely)
    gradient = cv.convertScaleAbs(gradient)

    # We expect the field to have a small gradient
    _, gradient_threshold = cv.threshold(gradient, 80, 255, cv.THRESH_BINARY_INV)

    # Opening
    opening = cv.morphologyEx(gradient_threshold, cv.MORPH_OPEN, np.ones((17,17), np.uint8))

    # Closing
    closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, np.ones((31,31), np.uint8))

    # The largest contour is considered the field
    contours, _ = cv.findContours(closing, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contour = sorted(contours, key=cv.contourArea)[-1]

    # Compute field
    field = np.zeros(img_gray.shape, np.uint8)
    field = cv.drawContours(field, [contour], -1, 255, -1)
    field = cv.dilate(field, np.ones((31,31), np.uint8)) # apply dilation to recover possible line losses with previous operations

    if debug:
        img_test1 = cv.cvtColor(img_gray, cv.COLOR_GRAY2BGR)
        img_test1 = cv.bitwise_and(img_test1, img_test1, mask=field)
        cv.imshow('Gradient', gradient)
        cv.imshow('Gradient Threshold', gradient_threshold)
        cv.imshow('Opening', opening)
        cv.imshow('Closing', closing)
        cv.imshow('Field', img_test1)

    return field



# Applies Canny detection followed by filtering using HoughP
def getCannyLines(img_gray, minWidth = 90, minBlackWhiteRatio = 5.0, offset = 25):
    canny = cv.Canny(img_gray, 30, 120)
    blur = cv.blur(canny, (3, 3))
    closed = cv.morphologyEx(canny, cv.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    lines = cv.HoughLinesP(blur, 1, np.pi / 180, 80, minLineLength=60, maxLineGap=10)
    lines = lines[:, 0, :] if lines is not None else []

    canny_lines = []
    for line in lines:
        x1, y1, x2, y2 = line
        
        rbox = cv.minAreaRect(np.array([(x1, y1), (x2, y2)], np.float32))
        x, y, width, height, angle = get_box_values(rbox)

        if angle == 0.0 or angle == 90.0:
            continue

        if width < minWidth:
            continue

        height += offset * 2
        rbox = ((x,y), (width, height), angle)
        pts = cv.boxPoints(rbox).astype(np.int32)

        mask = np.zeros(closed.shape, np.uint8)
        mask = cv.drawContours(mask, [pts], -1, 255, -1)
        img_roi = closed.copy()
        img_roi[mask == 0] = 120

        blacks = len(img_roi[img_roi == 0])
        whites = len(img_roi[img_roi == 255])

        if blacks == 0 or whites == 0: # noisy data
            continue

        if blacks * 1.0 / whites <= minBlackWhiteRatio: # there should be more black than white around the detected line
            continue

        canny_lines.append(line)
    canny_lines = np.array(canny_lines)
    
    if debug:
        img_test = cv.cvtColor(closed, cv.COLOR_GRAY2BGR)
        for x1,y1,x2,y2 in lines:
            rbox = cv.minAreaRect(np.array([(x1, y1), (x2, y2)], np.float32))
            x, y, width, height, angle = get_box_values(rbox) 
            height += offset * 2
            rbox = ((x,y), (width, height), angle)
            pts = cv.boxPoints(rbox).astype(np.int32)
            #cv.drawContours(img_test, [pts], -1, (0, 255, 0), 1, cv.LINE_AA)
            cv.line(img_test, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv.imshow('Canny', canny)
        cv.imshow('Canny Lines', img_test)

    return canny, closed, canny_lines



# Group together lines with similar 'm' and 'b' (y = m*x + b)
def groupLines(img, canny_lines, toleranceM = 0.2, toleranceB = 20):
    bins = []
    for x1, y1, x2, y2 in canny_lines:
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

    if debug:
        img_test = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        for _, _, lines in bins:
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            for x1, y1, x2, y2 in lines:
                m = (y2-y1)/(x2-x1)
                b = y1 - m*x1
                cv.line(img_test, (x1, y1), (x2, y2), color, 1)
        cv.imshow('Canny Lines Grouped', img_test)

    return bins



# Find best fit for each bin
def fitGroupedLines(canny, bins):
    fitted_lines = []
    for avg_m, avg_b, lines in bins:
        # Rearrange data to use in fitLine function.
        points = []
        for line in lines:
            x1, y1, x2, y2 = line
            points.append((x1, y1))
            points.append((x2, y2))
        points = np.array(points)

        # Find and store best fit
        regression = cv.fitLine(points, cv.DIST_L2, 0, 0.01, 0.01)[:,0]
        fitted_lines.append(regression)
    fitted_lines.sort(key=lambda line: line[1] / line[0])
    fitted_lines = np.array(fitted_lines)

    if debug:
        img_test = cv.cvtColor(canny, cv.COLOR_GRAY2BGR)
        for vx, vy, x0, y0 in fitted_lines:
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            x1, y1 = int(x0 - 1000*vx), int(y0 - 1000*vy)
            x2, y2 = int(x0 + 1000*vx), int(y0 + 1000*vy)
            cv.line(img_test, (x1,y1), (x2,y2), color, 2)
        cv.imshow('Fitted Lines', img_test)

    return fitted_lines

def findBestLines(canny, fitted_lines, toleranceM = 0.17, minPointDistance = 30):
    groups = []
    for line in fitted_lines:
        vx, vy, x0, y0 = line
        m = vy / vx

        i = 0
        while i < len(groups):
            (avg_vx, avg_vy), group_lines = groups[i]
            avg_m = avg_vy / avg_vx
            if should_group(groups[i], line, toleranceM, minPointDistance):
                group_lines.append(line)
                groups[i] = ( ((avg_vx + vx)/2, (avg_vy + vy)/2), group_lines )
                break
            i += 1
        if i == len(groups):
            groups.append( ((vx, vy), [line]) )

    maxCrossProduct = 0
    group1 = []
    group2 = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            u, _ = groups[i]
            v, _ = groups[j]
            cp = cross_product(u, v)

            if cp > maxCrossProduct:
                maxCrossProduct = cp
                group1 = groups[i]
                group2 = groups[j]

    if debug:
        img_test = cv.cvtColor(canny, cv.COLOR_GRAY2BGR)
        for _, lines in [group1, group2]:
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            for vx, vy, x0, y0 in lines:
                x1, y1 = int(x0 - 1000*vx), int(y0 - 1000*vy)
                x2, y2 = int(x0 + 1000*vx), int(y0 + 1000*vy)
                cv.line(img_test, (x1, y1), (x2, y2), color, 2)
                cv.circle(img_test, (x0, y0), 5, (0, 0, 255), -1)
        cv.imshow('Best Lines Grouped', img_test)

    return [group1, group2]



def findInterestPoints(canny, groups, imageOffset = 50):
    (_, lines1), (_, lines2) = groups

    img = canny.copy()
    img = cv.copyMakeBorder(img, 0, 50, 0, 50, cv.BORDER_CONSTANT, value=255)
    height, width = img.shape

    points = []
    for i in range(len(lines1)):
        for j in range(len(lines2)):
            a, c = vector2cartesian(lines1[i]) # y = a*x + c
            b, d = vector2cartesian(lines2[j]) # y = b*x + d

            if a == b: # lines are parallel
                continue

            x = (d-c) / (a-b)
            y = (a*d - b*c) / (a-b)

            if (x >= 0 and x <= width + imageOffset) and (y >= 0 and y <= height + imageOffset):
                points.append((x,y))

    if debug:
        img_copy = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        for x,y in points:
            cv.circle(img_copy, (int(x), int(y)), 5, (0, 255, 0), -1)
        cv.imshow('Interest Points', img_copy)

    return points



if __name__ == "__main__":
    for i in range(1, 4):
        filename = 'images/{}.png'.format(i)
        img, img_gray = loadImage(filename)

        field = getField(img_gray)

        # canny, closed, canny_lines = getCannyLines(img_gray)

        # bins = groupLines(closed, canny_lines)
        # fitted_lines = fitGroupedLines(canny, bins)
        # groups = findBestLines(canny, fitted_lines)
        # points = findInterestPoints(canny, groups)

        cv.waitKey(0)
    cv.destroyAllWindows()