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

    avg_m, group_lines = group
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



def loadImage(filename):
    img = cv.imread(filename)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    if debug:
        cv.imshow('Image', img)
        cv.imshow('Gray Image', img_gray)

    return img, img_gray



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
        for x1,y1,x2,y2 in canny_lines:
            rbox = cv.minAreaRect(np.array([(x1, y1), (x2, y2)], np.float32))
            x, y, width, height, angle = get_box_values(rbox) 
            height += offset * 2
            rbox = ((x,y), (width, height), angle)
            pts = cv.boxPoints(rbox).astype(np.int32)
            cv.drawContours(img_test, [pts], -1, (0, 255, 0), 1, cv.LINE_AA)
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

def findBestLines(canny, fitted_lines, toleranceM = 0.148, minPointDistance = 30):
    groups = []

    for line in fitted_lines:
        vx, vy, x0, y0 = line
        m = vy / vx

        i = 0
        while i < len(groups):
            avg_m, group_lines = groups[i]
            if should_group(groups[i], line, toleranceM, minPointDistance):
                group_lines.append(line)
                groups[i] = ((avg_m+m)/2, group_lines)
                break
            i += 1
        if i == len(groups):
            groups.append((m, [line]))

    groups = np.array(groups)

    if debug:
        img_test = cv.cvtColor(canny, cv.COLOR_GRAY2BGR)
        for avg_m, lines in groups:
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            for vx, vy, x0, y0 in lines:
                x1, y1 = int(x0 - 1000*vx), int(y0 - 1000*vy)
                x2, y2 = int(x0 + 1000*vx), int(y0 + 1000*vy)
                cv.line(img_test, (x1, y1), (x2, y2), color, 2)
                cv.circle(img_test, (x0, y0), 5, (0, 0, 255), -1)
        cv.imshow('Best Lines Grouped', img_test)



if __name__ == "__main__":
    for i in range(1, 4):
        filename = 'images/{}.png'.format(i)
        img, img_gray = loadImage(filename)

        canny, closed, canny_lines = getCannyLines(img_gray)

        bins = groupLines(closed, canny_lines)
        fitted_lines = fitGroupedLines(canny, bins)
        best_lines = findBestLines(canny, fitted_lines)

        cv.waitKey(0)
    cv.destroyAllWindows()