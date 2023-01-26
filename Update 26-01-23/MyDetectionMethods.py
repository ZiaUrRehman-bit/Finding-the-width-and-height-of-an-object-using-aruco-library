import cv2 as cv
import numpy as np

class MyDetectionMethod():
    def __init__(self):
        pass

    def detectUsingThresh(self, img, threshVal = 140):
        imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        invertGray = cv.bitwise_not(imgGray)
        ret, binaryThresh = cv.threshold(invertGray, threshVal, 255, cv.THRESH_BINARY)

        contours, hierarchy = cv.findContours(binaryThresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

        objectContours = []

        for cont in contours:
            area = cv.contourArea(cont)

            if 40000 > area > 3000:
                objectContours.append(cont)

        return objectContours

    def detectUsingCanny(self, img, lower= 160,upper= 200):
        # Blur img for removing noise
        blurred_img = cv.GaussianBlur(img,(5,5),0)

        # Apply canny edge detection
        cannyimg = cv.Canny(blurred_img, lower, upper)
        dilateKernel = np.ones((5,5), "uint8")

        imgDilation = cv.dilate(cannyimg, dilateKernel, iterations=1)
        contours, hierarchy = cv.findContours(imgDilation, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        cv.imshow("dilat", imgDilation)

        objectContours = []

        for cont in contours:
            area = cv.contourArea(cont)

            if 20000 > area > 18000:
                 objectContours.append(cont)
            if 3000 > area > 2000:
                 objectContours.append(cont)
            # objectContours.append(cont)

        return objectContours

#Both Canny and threshold are used to detect edges and count the number of objects
#thesholding is not so smooth like Canny but I think thresholding is faster to detect edges anyway

