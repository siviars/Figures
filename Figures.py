import math
import cv2
import numpy as np

shapes = 0

img = cv2.imread('Objects/png_image.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(gray, 50, 255, 0)

contours, hierarchy = cv2.findContours(thresh, 1, 2)

circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 2.3, 100)


def get_angle(point_1, point_2, point_3):
    m1 = (point_2[0][1] - point_1[0][1]) / (point_2[0][0] - point_1[0][0])
    m2 = (point_3[0][1] - point_1[0][1]) / (point_3[0][0] - point_1[0][0])
    ang_r = math.atan((m2 - m1) / (1 + (m2 * m1)))
    ang_d = math.ceil(math.degrees(ang_r))
    return ang_d


def get_edge(point_1, point_2):
    if point_1[0][1] != point_2[0][1]:
        return math.sqrt(
            (point_2[0][1] - point_1[0][1]) ** 2 + (point_2[0][0] - point_1[0][0]) ** 2)
    return point_2[0][0] - point_1[0][0]


if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        shapes = shapes + 1
        cv2.circle(img, (x, y), r, (225, 0, 0), 3)
        cv2.putText(img, 'Circle radius ' + str(r), (x - r - 20, y - r - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0),
                    2)

for cnt in contours:
    x1, y1 = cnt[0][0]
    approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
    if len(approx) == 4:
        x, y, w, h = cv2.boundingRect(cnt)
        ratio = float(w) / h
        if 0.9 <= ratio <= 1.1:
            shapes = shapes + 1
            edge = round(get_edge(approx[0], approx[1]))
            img = cv2.drawContours(img, [cnt], -1, (0, 255, 0), 3)
            cv2.putText(img, 'Square edge ' + str(edge), (x1 - 20, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0),
                        2)

    if len(approx) == 3:
        shapes = shapes + 1
        pt1, pt2, pt3 = approx
        angle = get_angle(pt1, pt2, pt3)
        img = cv2.drawContours(img, [cnt], -1, (0, 0, 255), 3)
        cv2.putText(img, 'Triangle angle ' + str(angle), (x1 - 20, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 2)

cv2.putText(img, 'Total shapes - ' + str(shapes), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (0, 0, 0), 2)
cv2.imshow("Shapes", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
