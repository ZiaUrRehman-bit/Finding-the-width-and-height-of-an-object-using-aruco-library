
import cv2 as cv
import numpy as np

class MyDetectionMethods():
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

            if area > 2000:
                objectContours.append(cont)

        return objectContours

    def detectUsingCanny(self, img, lower= 120,upper= 160):
        # Blur the image to remove noise
        blurred_image = cv.GaussianBlur(img,(5,5),0)

        # Apply canny edge detection
        cannyImge = cv.Canny(blurred_image, lower, upper)
        dilateKernel = np.ones((5,5), "uint8")

        imgDilation = cv.dilate(cannyImge, dilateKernel, iterations=1)
        contours, hierarchy = cv.findContours(imgDilation, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        cv.imshow("dilat", imgDilation)

        objectContours = []

        for cont in contours:
            area = cv.contourArea(cont)

            if area > 2000:
                objectContours.append(cont)

        return objectContours


# img = cv.imread("2.jpeg")

# detectObject = MyDetectionMethods()

# # print(img.shape)
# img = cv.resize(img, (400, 550))
# copyImg1 = img.copy()
# copyImg2 = img.copy()

# contours = detectObject.detectUsingThresh(img, threshVal=110)
# cv.drawContours(copyImg1, contours, -1, (255, 0, 255), 3)

# imgd, contours2 = detectObject.detectUsingCanny(img)
# cv.drawContours(copyImg2, contours2, -1, (255, 0, 255), 3)


# cv.imshow("ContoursUsingThreshold", copyImg1)
# cv.imshow("ContoursUsingCanny", copyImg2)
# cv.imshow("cann", imgd)
# cv.waitKey()

# cv.destroyAllWindows()