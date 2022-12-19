# in this projrct we are going to learn how to perform object measurement using Opencv

import cv2 as cv
import numpy as np
import utilis

webcame = False
cam = cv.VideoCapture(1)
path = "1.jpeg"
# cam.set(10, 160)  # this will set the brightness of camera
cam.set(3, 1020)  # width
cam.set(4, 680)  # height

while True:
    if webcame: Success, frame = cam.read()
    else: frame = cv.imread(path)

    frame, contours = utilis.findCountoures(frame, show=True,
                                                    minArea=50000, filter=4)

    if len(contours) != 0: 
        biggest = contours[0][2]
        # print(biggest)

        

    frame = cv.resize(frame, (0,0), None, 0.5, 0.5)
    cv.imshow("frame", frame)
    k = cv.waitKey(1)

    if k == ord("q"):
        cv.destroyAllWindows()
        break