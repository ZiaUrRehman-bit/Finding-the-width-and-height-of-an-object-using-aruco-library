import cv2 as cv
import numpy as np

# find contours
def findCountoures(frame, cThr = [100, 100], show = True, minArea = 1000, filter=0, draw=False):
    
    grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blurFrame = cv.GaussianBlur(grayFrame, (5,5), 1)
    cannyFrame = cv.Canny(blurFrame,cThr[0], cThr[1])

    kernel = np.ones((5,5))

    dilateFrame = cv.dilate(cannyFrame, kernel, iterations=3)
    thrsFrame = cv.erode(dilateFrame, kernel, iterations=2)

    if show: cv.imshow("canny", thrsFrame)

    contours, hiearchy = cv.findContours(thrsFrame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    finalContours = []
    for i in contours:
        area = cv.contourArea(i)

        if area > minArea:
            parimeter = cv.arcLength(i, True)
            approx = cv.approxPolyDP(i, 0.02*parimeter, True)

            bbox = cv.boundingRect(approx)

            if filter > 0:
                if len(approx) == filter:
                    finalContours.append([len(approx), area, approx, bbox, i])
            else:
                finalContours.append([len(approx), area, approx, bbox, i])
    
    finalContours = sorted(finalContours, key = lambda x:x[1], reverse=True)

    if draw:
        for con in finalContours:
            cv.drawContours(frame, con[4], -1, (255,0,255), 3)
    
    return frame, finalContours