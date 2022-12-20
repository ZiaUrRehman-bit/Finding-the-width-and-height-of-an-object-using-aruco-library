import cv2 as cv
import numpy as np
from MyDetectionMethods import MyDetectionMethods

param = cv.aruco.DetectorParameters_create()
aruco_dict = cv.aruco.Dictionary_get(cv.aruco.DICT_5X5_50)


img = cv.imread("2.jpeg")
img = cv.resize(img, (400, 550))

contours = MyDetectionMethods.detectUsingCanny('self', img=img, lower=130, upper=200)
# cv.drawContours(img, contours, -1, (255, 0, 255), 3)

for cont in contours:

    result = cv.minAreaRect(cont)
    print(result)
    (x, y), (w, h), angle = result
    #
    cv.circle(img, (int(x),int(y)),2, (0,255,255),2)
    #
    boundingBox = cv.boxPoints(result)
    # #convert to integer
    boundingBox = np.int0(boundingBox)
    print(boundingBox[0])
    #
    # # draw rectangle
    cv.line(img, (boundingBox[0][0], boundingBox[0][1]), (boundingBox[1][0],boundingBox[1][1]), (0,0,255),2)
    cv.line(img, (boundingBox[0][0], boundingBox[0][1]), (boundingBox[3][0],boundingBox[3][1]), (0,0,255),2)
    cv.line(img, (boundingBox[1][0],boundingBox[1][1]), (boundingBox[2][0],boundingBox[2][1]), (0,0,255),2)
    cv.line(img, (boundingBox[3][0],boundingBox[3][1]), (boundingBox[2][0],boundingBox[2][1]), (0,0,255),2)
    #
    # # print(boundingBox)
    #
    # # print height and width
    cv.putText(img, f'width: {round(w, 2)}', (int(x-50),int(y-30)),
                cv.FONT_HERSHEY_PLAIN, 1, (255,0,255), 1)
    cv.putText(img, f'height: {round(h, 2)}', (int(x-50),int(y-5)),
                cv.FONT_HERSHEY_PLAIN, 1, (255,0,255), 1)

cv.imshow("image", img)
cv.waitKey()

cv.destroyAllWindows()