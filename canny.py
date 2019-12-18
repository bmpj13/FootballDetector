import cv2 as cv
import numpy as np
import random
import math

debug = True



''' Helpers '''

def get_box_values(box):
    (x, y), (width, height), angle = box
    x, y, width, height = x, y, width, height

    if angle < -45.0:     # Switch height and width if angle between -90 and -45 ( width, height 90º = height, width 0º )
        width, height = height, width

    return x, y, width, height, angle

def bounding_box_info(img, pts):
    mask = np.zeros(img.shape, np.uint8)
    mask = cv.drawContours(mask, [pts], -1, 255, -1)
    img_roi = img.copy()
    img_roi[mask == 0] = 128

    return len(img_roi[img_roi == 0]), len(img_roi[img_roi == 255])

def cross_product(vec1, vec2):
    (x1, y1) = vec1
    (x2, y2) = vec2
    return (x1 * y2) - (x2 * y1)

def vector2cartesian(line):
    vx, vy, x0, y0 = line
    slope = vy / vx
    axisy = y0 - slope * x0
    return slope, axisy

def points2cartesian(line):
    x1, y1, x2, y2 = line
    m = (y2-y1)/(x2-x1)
    b = y1 - m*x1
    return m, b

def increment_average(avg, count, value):
    # https://math.stackexchange.com/questions/106700/incremental-averageing
    return avg + ((value - avg) / count)

def find_best_group(cartesian_line, groups, toleranceM, toleranceB):
    in_tolerance = lambda l,g: abs(l[0] - g[0]) < toleranceM and abs(l[1] - g[1]) < toleranceB

    m, b = cartesian_line

    index = -1
    min_m_diff = 9999
    for i, group in enumerate(groups):
        (avg_m, avg_b), _ = group

        if in_tolerance((m, b), (avg_m, avg_b)):
            m_diff = abs(m - avg_m)
            
            if m_diff < min_m_diff:
                min_m_diff = m_diff
                index = i

    return index != -1, index

def no_intersection(line, group_lines, width, height, distance):
    for gl in group_lines:
        a, c = vector2cartesian(line)
        b, d = vector2cartesian(gl)

        if a == b:  # parallel
            return True

        x = (d-c) / (a-b)
        y = (a*d - b*c) / (a-b)

        if (x >= -distance and x <= width + distance) and (y >= -distance and y <= height + distance):
            return False
    return True

def find_best_fit_group(shape, line, groups, toleranceM, minIntersectDistance):
    in_tolerance = lambda l,g: abs(l - g) < toleranceM

    height, width = shape
    m, b = vector2cartesian(line)

    index = -1
    min_m_diff = 9999
    for i, group in enumerate(groups):
        (avg_vx, avg_vy), group_lines = group
        avg_m = avg_vy / avg_vx

        if in_tolerance(m, avg_m) and no_intersection(line, group_lines, width, height, minIntersectDistance):
            m_diff = abs(m - avg_m)
            
            if m_diff < min_m_diff:
                min_m_diff = m_diff
                index = i

    return index != -1, index

def get_point_id(v_id, h_id, v_lines, h_lines):
    if len(v_lines) != 3 or len(h_lines) != 4:
        return None

    return h_id * 3 + v_id + 1




''' API '''

def loadImage(filename):
    img = cv.imread(filename)
    img = cv.resize(img, (800, 400), interpolation=cv.INTER_AREA)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    if debug:
        cv.imshow('Image', img)
        cv.imshow('Gray Image', img_gray)

    return img, img_gray



def getField(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # Green HSV: (60, 255, 255)
    lower_green = np.array([30, 50, 50])
    upper_green = np.array([80, 255, 255])
    mask = cv.inRange(hsv, lower_green, upper_green)

    # The largest contour is considered the field
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contour = sorted(contours, key=cv.contourArea)[-1]

    # Compute field
    field = np.zeros(img.shape[0:2], np.uint8)
    field = cv.drawContours(field, [contour], -1, 255, -1)

    if debug:
        img_debug = cv.bitwise_and(img, img, mask=field)
        cv.imshow('Green Filter', mask)
        cv.imshow('Field', img_debug)

    return mask, field



# Applies Canny detection followed by filtering using HoughP
def getCannyLines(img_gray, field, boundingHeight = 3, minSupport = 0.9, minBlackWhiteRatio = 1.5):
    img = cv.bitwise_and(img_gray, img_gray, mask=field)
    canny = cv.Canny(img, 30, 100)
    closed = cv.morphologyEx(canny, cv.MORPH_CLOSE, np.ones((boundingHeight, boundingHeight), np.uint8))

    lines = cv.HoughLinesP(canny, 1, np.pi / 180, 40, minLineLength=50, maxLineGap=10)
    lines = lines[:, 0, :] if lines is not None else []

    canny_lines = []
    for line in lines:
        x1, y1, x2, y2 = line
        
        rbox = cv.minAreaRect(np.array([(x1, y1), (x2, y2)], np.float32))
        x, y, width, _, angle = get_box_values(rbox)
        if angle == 0.0 or angle == 90.0 or width == 0:
            continue
        
        height = boundingHeight
        pts1 = cv.boxPoints(((x,y), (width, height), angle)).astype(np.int32)
        _, whites = bounding_box_info(closed, pts1)
        support = whites / (width * height)

        if support <= minSupport:
            continue

        height *= 4
        pts2 = cv.boxPoints(((x,y), (width, height), angle)).astype(np.int32)
        blacks, whites = bounding_box_info(closed, pts2)
        black_white_ratio = blacks / whites
        
        if black_white_ratio < minBlackWhiteRatio:
            continue

        canny_lines.append(line)
    canny_lines = np.array(canny_lines)
    
    if debug:
        img_test = cv.cvtColor(closed, cv.COLOR_GRAY2BGR)
        for line in canny_lines:
            x1, y1, x2, y2 = line
            cv.line(img_test, (x1, y1), (x2, y2), (0, 0, 255), thickness=1)
        cv.imshow('Canny', canny)
        cv.imshow('Canny Lines', img_test)

    return canny, closed, canny_lines



# Group together lines with similar 'm' and 'b' (y = m*x + b)
def groupLines(img, canny_lines, toleranceM = 0.1, toleranceB = 10):
    groups = []
    for line in canny_lines:
        x1, y1, x2, y2 = line
        if x2 - x1 == 0:
            continue

        m, b = points2cartesian(line)
        found, index = find_best_group((m, b), groups, toleranceM, toleranceB)

        if found:
            (avg_m, avg_b), group_lines = groups[index]
            n = len(group_lines)
            avg_m = increment_average(avg_m, n, m)
            avg_b = increment_average(avg_b, n, b)
            group_lines.append(line)
            groups[index] = ((avg_m, avg_b), group_lines)
        else:
            groups.append(((m, b), [line]))

    groups = np.array(groups)

    if debug:
        img_test = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        for _, lines in groups:
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            for x1, y1, x2, y2 in lines:
                m = (y2-y1) / (x2-x1)
                b = y1 - m*x1
                cv.line(img_test, (x1, y1), (x2, y2), color, 1)
        cv.imshow('Canny Lines Grouped', img_test)

    return groups



# Find best fit for each bin
def fitGroupedLines(canny, bins):
    fitted_lines = []
    for _, lines in bins:
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


def group_by_index(index, fitted_lines, groups, shape, toleranceM, minIntersectDistance):
    line = fitted_lines[index]
    vx, vy, x0, y0 = line
    found, index = find_best_fit_group(shape, line, groups, toleranceM, minIntersectDistance)

    if found:
        (avg_vx, avg_vy), group_lines = groups[index]
        n = len(group_lines)
        avg_vx = increment_average(avg_vx, n, vx)
        avg_vy = increment_average(avg_vy, n, vy)
        group_lines.append(line)
        groups[index] = ((avg_vx, avg_vy), group_lines)
    else:
        groups.append(((vx, vy), [line]))

def findBestGroups(canny, fitted_lines, toleranceM = 0.3, minIntersectDistance = 500):
    # Group lines according to their slope and minimum intersection distance
    # We use a greedy approach, which might lead to erros. Since the the 'fitted_lines' are ordered by slope,
    # we compute the group for the first and last line at every iteration (which reduces the error - a bit...)
    groups = []
    for i in range(int(len(fitted_lines) / 2)):
        group_by_index(i, fitted_lines, groups, canny.shape, toleranceM, minIntersectDistance)
        group_by_index(len(fitted_lines)-1-i, fitted_lines, groups, canny.shape, toleranceM, minIntersectDistance)
    
    if len(fitted_lines) % 2 != 0:
        group_by_index(int(len(fitted_lines) / 2) + 1, fitted_lines, groups, canny.shape, toleranceM, minIntersectDistance)

    # Filter groups that only have one line
    filtered_groups = []
    for group in groups:
        _, group_lines = group

        if len(group_lines) > 1:
            filtered_groups.append(group)

    # Choose the lines that have a maximum cross product
    maxCrossProduct = 0
    group1 = []
    group2 = []
    for i in range(len(filtered_groups)):
        for j in range(i + 1, len(filtered_groups)):
            u, _ = filtered_groups[i]
            v, _ = filtered_groups[j]
            cp = cross_product(u, v)

            if cp > maxCrossProduct:
                maxCrossProduct = cp
                group1 = filtered_groups[i]
                group2 = filtered_groups[j]

    if debug:
        img_test = cv.cvtColor(canny, cv.COLOR_GRAY2BGR)
        for _, lines in groups:
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            for vx, vy, x0, y0 in lines:
                x1, y1 = int(x0 - 1000*vx), int(y0 - 1000*vy)
                x2, y2 = int(x0 + 1000*vx), int(y0 + 1000*vy)
                cv.line(img_test, (x1,y1), (x2,y2), color, 2)
        cv.imshow('Grouped Fitted Lines', img_test)

        img_test = cv.cvtColor(canny, cv.COLOR_GRAY2BGR)
        for _, lines in filtered_groups:
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            for vx, vy, x0, y0 in lines:
                x1, y1 = int(x0 - 1000*vx), int(y0 - 1000*vy)
                x2, y2 = int(x0 + 1000*vx), int(y0 + 1000*vy)
                cv.line(img_test, (x1,y1), (x2,y2), color, 2)
        cv.imshow('Filtered Grouped Fitted Lines', img_test)

        img_test = cv.cvtColor(canny, cv.COLOR_GRAY2BGR)
        for v, lines in [group1, group2]:
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            for vx, vy, x0, y0 in lines:
                x1, y1 = int(x0 - 1000*vx), int(y0 - 1000*vy)
                x2, y2 = int(x0 + 1000*vx), int(y0 + 1000*vy)
                cv.line(img_test, (x1, y1), (x2, y2), color, 2)
        cv.imshow('Best Lines Grouped', img_test)

    return [group1, group2]



def getScenarioInfo(canny, best_groups):
    (vx1, vy1), lines1 = best_groups[0]
    (vx2, vy2), lines2 = best_groups[1]

    # Horizontal lines should have higher variation in the 'x' axis
    if vx1 > vx2:
        horizontal = ((vx1, vy1), lines1)
        vertical = ((vx2, vy2), lines2)
    else:
        vertical = ((vx1, vy1), lines1)
        horizontal = ((vx2, vy2), lines2)

    # If variation in 'y' is positive, we're on the field's left side
    if horizontal[0][1] > 0:
        field_side = 'left'
    else:
        field_side = 'right'

    # Order lines
    if field_side == 'left':
        horizontal[1].sort(key=lambda l: l[3])
        vertical[1].sort(key=lambda l: l[2])
    else:
        horizontal[1].sort(key=lambda l: -l[3])
        vertical[1].sort(key=lambda l: -l[2])

    # Remove possible final outliers
    (v, lines) = vertical
    if len(lines) > 3:  # expected only 3 vertical lines
        last = 3 - len(lines)
        lines = lines[:last]
        vertical = (v, lines)

    (v, lines) = horizontal
    if len(lines) > 4:  # expected only 4 horizontal lines
        idx = 4 - len(lines)
        if field_side == 'left': lines = lines[abs(idx):]
        else: lines = lines[:idx]
        horizontal = (v, lines)

    if debug:
        img_test = cv.cvtColor(canny, cv.COLOR_GRAY2BGR)
        for vx, vy, x0, y0 in vertical[1]:
            x1, y1 = int(x0 - 1000*vx), int(y0 - 1000*vy)
            x2, y2 = int(x0 + 1000*vx), int(y0 + 1000*vy)
            cv.line(img_test, (x1, y1), (x2, y2), (0, 0, 255), 2)
        for vx, vy, x0, y0 in horizontal[1]:
            x1, y1 = int(x0 - 1000*vx), int(y0 - 1000*vy)
            x2, y2 = int(x0 + 1000*vx), int(y0 + 1000*vy)
            cv.line(img_test, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv.imshow('Final lines', img_test)

    return field_side, vertical, horizontal



def findInterestPoints(canny, vertical, horizontal):
    (_, lines_vertical) = vertical
    (_, lines_horizontal) = horizontal
    height, width = canny.shape

    points = []
    for i in range(len(lines_vertical)):
        for j in range(len(lines_horizontal)):
            a, c = vector2cartesian(lines_vertical[i]) # y = a*x + c
            b, d = vector2cartesian(lines_horizontal[j]) # y = b*x + d

            if a == b: # lines are parallel
                continue

            x = (d-c) / (a-b)
            y = (a*d - b*c) / (a-b)

            if (x >= 0 and x <= width) and (y >= 0 and y <= height):
                point_id = get_point_id(i, j, lines_vertical, lines_horizontal)
                points.append((point_id, (x,y)))

    if debug:
        img_copy = cv.cvtColor(canny, cv.COLOR_GRAY2BGR)
        for point_id, (x,y) in points:
            cv.circle(img_copy, (int(x), int(y)), 5, (0, 255, 0), -1)
            cv.imshow('Interest Points', img_copy)

    return points

def getPlayersMask(field, greens_mask):
    players = cv.bitwise_not(greens_mask)
    players = cv.bitwise_and(players, players, mask=field)

    players_erode = cv.erode(players, np.ones((7,1), np.uint8))
    players_dilate = cv.dilate(players_erode, np.ones((9,3), np.uint8))

    if debug:
        cv.imshow('Players Mask', players)
        cv.imshow('Players Eroded', players_erode)
        cv.imshow('Players Dilated', players_dilate)

    return players_dilate

def compute(filename, use_debug=False):
    global debug
    debug = use_debug
    img, img_gray = loadImage(filename)

    greens_mask, field = getField(img)
    # canny, closed, canny_lines = getCannyLines(img_gray, field)
    # groups = groupLines(closed, canny_lines)
    # fitted_lines = fitGroupedLines(canny, groups)
    # best_groups = findBestGroups(canny, fitted_lines)
    # _, vertical, horizontal = getScenarioInfo(canny, best_groups)
    # points = findInterestPoints(canny, vertical, horizontal)
    # players = getPlayersMask(field, greens_mask)

    # return img, points, players, field


if __name__ == "__main__":
    for i in range(1, 6):
        filename = 'images/{}.png'.format(i)
        compute(filename, True)
        cv.waitKey(0)
    cv.destroyAllWindows()