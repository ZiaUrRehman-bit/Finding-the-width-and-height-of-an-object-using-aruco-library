# Finding-the-width-and-height-of-an-object-using-aruco-library
#### cv2.VideoCapture
cv2.VideoCapture is a class in the cv2 (OpenCV) library in Python that allows you to capture video from a file or a device, such as a camera. It is a powerful tool for reading video streams and can be used to process video frames in real-time.

##### To use cv2.VideoCapture

 1. you first need to import the cv2 library.

 2. Then, you can create a cv2.VideoCapture object 

 3. and use its open() method to specify the source of the video, which can be a file path or a device index (for example, 0 for the default camera). 

 4. Once the video source is open, you can use the read() method to read and process each frame of the video 
 
 5. and the release() method and destroyAllWindows() method to close the video stream and all windows when you are done.

Here is an example of how to use cv2.VideoCapture to read and display video from a file:
import cv2

# Create a cv2.VideoCapture object
cap = cv2.VideoCapture(0) # for webcame write 0 

# Check if the video was opened successfully
if not cap.isOpened():
    print("Error opening video file")

# Read and process each frame of the video
while cap.isOpened():
    # Read the next frame , return two things 1. frame, 2. boolean value(True or False)---> True when frame is 
    # successfully read, False when frame is not successfully read
    ret, frame = cap.read()

    # Check if we reached the end of the video
    if ret == False:
        break

    # Display the frame
    cv2.imshow("Frame", frame)

    # Wait for a key press
    key = cv2.waitKey(1)

    # if 'q' pressed then break the loop exit from it
    if key == ord("q"):
        break

# Release the video stream and close all windows
cap.release()
cv2.destroyAllWindows()

### Here is question rises, why we need while loop in above code?

Let's write the same code written above without loop and see what happend 
import cv2

# Create a cv2.VideoCapture object
cap = cv2.VideoCapture("zia.mp4")

# Check if the video was opened successfully
if not cap.isOpened():
    print("Error opening video file")

# # Read and process each frame of the video
# while cap.isOpened():
# Read the next frame , return two things 1. frame, 2. boolean value(True or False)---> True when frame is 
# successfully read, False when frame is not successfully read
ret, frame = cap.read()

# Check if we reached the end of the video
if not ret:
    print("frame is not readed successfully")

# Display the frame
cv2.imshow("Frame", frame)

# Wait for a key press
key = cv2.waitKey()

# if 'q' pressed then break the loop exit from it
if key == ord("q"):
    
    # Release the video stream and close all windows
    cap.release()
    cv2.destroyAllWindows()
In Frame window, you observe that there is only first frame from video is captured and readed and then shown.

So, in order to read all frames from video or webcam you need a loop
### How we acces mobile camera

1. You need to download & install Iriun Webcam application in android mobile phone( for ios, this is also available)
    link: https://play.google.com/store/apps/details?id=com.jacksoftw.webcam&hl=en&gl=US&pli=1

2. Now you need to download & install Iriun Webcam software in your PC or laptop, link: https://iriun.com/

The code for this is remains same as used for accessing webcam just give the argument to cv2.VideoCapture(1)
import cv2

# Create a cv2.VideoCapture object
cap = cv2.VideoCapture(1) # for mobile camera in my case address is 1

# Check if the video was opened successfully
if not cap.isOpened():
    print("Error opening video file")

# Read and process each frame of the video
while cap.isOpened():
    # Read the next frame , return two things 1. frame, 2. boolean value(True or False)---> True when frame is 
    # successfully read, False when frame is not successfully read
    ret, frame = cap.read()

    # Check if we reached the end of the video
    if not ret:
        break

    # Display the frame
    cv2.imshow("Frame", frame)

    # Wait for a key press
    key = cv2.waitKey(1)

    # if 'q' pressed then break the loop exit from it
    if key == ord("q"):
        break

# Release the video stream and close all windows
cap.release()
cv2.destroyAllWindows()

### cv2.cirlce

cv2.circle() is a function in the cv2 (OpenCV) library in Python that allows you to draw a circle on an image. 

It takes the following arguments:

1. image: A numpy array representing the image on which you want to draw the circle.

2. center: A tuple representing the center of the circle, given as (x, y) coordinates.

3. radius: The radius of the circle.

4. color: The color of the circle, given as a tuple of 3 or 4 values representing the blue, green, red, and optionally alpha channels.

5. thickness: The thickness of the circle outline, in pixels. If set to -1, the circle will be filled.

Here is an example of how to use cv2.circle() to draw a red, filled circle on an image:


import cv2
import numpy as np

# Create a blank image
image = np.zeros((400, 400, 3), dtype=np.uint8)

width = 200
height = 20
radius = 50

# Draw a red, filled circle on the image
cv2.circle(image, (width, height), radius, (0, 0, 255), 6)  # -1 for filled the cirlce

# Display the image
cv2.imshow("Circle", image)
k = cv2.waitKey(0)

if k == ord("q"):
    cv2.destroyAllWindows()

cv2.destroyAllWindows()
You can also use cv2.circle() to draw multiple circles on an image by calling it multiple times with different arguments. 

For example:
import cv2
import numpy as np

# Create a blank image
image = np.zeros((400, 400, 3), dtype=np.uint8)

# Draw a red circle on the image
cv2.circle(image, (100, 100), 50, (0, 0, 255), -1)

# Draw a green circle on the image
cv2.circle(image, (300, 100), 50, (0, 255, 0), -1)

# Draw a blue circle on the image
cv2.circle(image, (200, 300), 50, (255, 0, 0), -1)

# Display the image
cv2.imshow("Circles", image)
k = cv2.waitKey()

if k == ord("q"):
    cv2.destroyAllWindows()

cv2.destroyAllWindows()
### Now draw the circle in side the video
import cv2

# Create a cv2.VideoCapture object
cap = cv2.VideoCapture("zia.mp4")

# Check if the video was opened successfully
if not cap.isOpened():
    print("Error opening video file")

# Read and process each frame of the video
while cap.isOpened():
    # Read the next frame , return two things 1. frame, 2. boolean value(True or False)---> True when frame is 
    # successfully read, False when frame is not successfully read
    ret, frame = cap.read()

    # Check if we reached the end of the video
    if not ret:
        break

    # Draw a blue circle on the image
    cv2.circle(frame, (200, 300), 50, (255, 0, 0), -1) 

    # Display the frame
    cv2.imshow("Frame", frame)

    # Wait for a key press
    key = cv2.waitKey(1)

    # if 'q' pressed then break the loop exit from it
    if key == ord("q"):
        break

# Release the video stream and close all windows
cap.release()
cv2.destroyAllWindows()

### what is cv2.rectangle(), and how it can used?

cv2.rectangle() is a function in the OpenCV (Open Computer Vision) library, a popular library for image processing tasks in Python. It can be used to draw a rectangle on an image.

The cv2.rectangle() function takes the following arguments:

1. image: the image on which to draw the rectangle. This should be a 2D NumPy array with dimensions (height, width, channels).

2. (10, 10) and (100, 100): these are the coordinates of the top-left and bottom-right corner of the rectangle, respectively.
    x , y         w , h
3. (255, 0, 0): this is the color of the rectangle, specified as a tuple of (B, G, R) values.

4. 2: this is the thickness of the rectangle outline, in pixels. If you set this to a value of -1, the rectangle will be filled with the specified color.
import cv2

# Load an image
image = cv2.imread('1.jpeg')

# Draw a rectangle on the image
cv2.rectangle(image, (10, 10), (100, 100), (255, 0, 0), -1)  

# Display the image
cv2.imshow('Image with rectangle', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

### Now draw the rectangle inside the video or webcam
import cv2

# Create a cv2.VideoCapture object
cap = cv2.VideoCapture("zia.mp4")

# Check if the video was opened successfully
if not cap.isOpened():
    print("Error opening video file")

# Read and process each frame of the video
while cap.isOpened():
    # Read the next frame , return two things 1. frame, 2. boolean value(True or False)---> True when frame is 
    # successfully read, False when frame is not successfully read
    ret, frame = cap.read()

    # Check if we reached the end of the video
    if not ret:
        break

    # Draw a rectangle on the frame
    cv2.rectangle(frame, (10, 10), (100, 100), (255, 0, 0), -1)  

    # Draw a rectangle on the frame
    cv2.rectangle(frame, (200, 200), (400, 400), (255, 0, 0), -1) 

    # Display the frame
    cv2.imshow("Frame", frame)

    # Wait for a key press
    key = cv2.waitKey(1)

    # if 'q' pressed then break the loop exit from it
    if key == ord("q"):
        break

# Release the video stream and close all windows
cap.release()
cv2.destroyAllWindows()

### what is cv2.line(), and how it can used?

cv2.line() is a function in the OpenCV (Open Computer Vision) library that can be used to draw a line on an image.

The cv2.line() function takes the following arguments:

1. image: the image on which to draw the line. This should be a 2D NumPy array with dimensions (height, width, channels).

2. (10, 10) and (100, 100): these are the coordinates of the start and end points of the line, respectively.
    x , y         w , h
    
3. (255, 0, 0): this is the color of the line, specified as a tuple of (B, G, R) values.

4. 2: this is the thickness of the line, in pixels.
import cv2

# Load an image
image = cv2.imread('1.jpeg')

# Draw a line on the image
cv2.line(image, (10, 10), (10, 100), (255, 0, 0), 2)

# Display the image
cv2.imshow('Image with line', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

### Now draw the line iside the video or webcam
import cv2

# Create a cv2.VideoCapture object
cap = cv2.VideoCapture("zia.mp4")

# Check if the video was opened successfully
if not cap.isOpened():
    print("Error opening video file")

# Read and process each frame of the video
while cap.isOpened():
    # Read the next frame , return two things 1. frame, 2. boolean value(True or False)---> True when frame is 
    # successfully read, False when frame is not successfully read
    ret, frame = cap.read()

    # Check if we reached the end of the video
    if not ret:
        break

    # Draw a line on the frame
    cv2.line(frame, (10, 10), (100, 100), (255, 200, 0), 2)

    # Display the frame
    cv2.imshow("Frame", frame)

    # Wait for a key press
    key = cv2.waitKey(1)

    # if 'q' pressed then break the loop exit from it
    if key == ord("q"):
        break

# Release the video stream and close all windows
cap.release()
cv2.destroyAllWindows()

### what is cv2.putText(), and how it can used?

cv2.putText() is a function in the OpenCV (Open Computer Vision) library that can be used to draw text on an image.

The cv2.putText() function takes the following arguments:

1. image: the image on which to draw the text. This should be a 2D NumPy array with dimensions (height, width, channels).

2. 'Hello, World!': this is the text to draw on the image.

3. (10, 50): this is the coordinate of the bottom-left corner of the text.

4. font: this is the font to use for the text. In this example, the cv2.FONT_HERSHEY_SIMPLEX font is used.

5. 1: this is the font scale. A value of 1 means that the text will be rendered at the original size.

6. (255, 255, 255): this is the color of the text, specified as a tuple of (B, G, R) values.

7. 2: this is the thickness of the text, in pixels.

8. cv2.LINE_AA: this is an optional parameter that specifies the type of line to use when rendering the text. cv2.LINE_AA stands for "anti-aliased line", which means that the text will be drawn smoothly.
import cv2

# Load an image
image = cv2.imread('1.jpeg')

# Define the font to use
font = cv2.FONT_HERSHEY_COMPLEX

# Draw text on the image
cv2.putText(image, 'Hello, World!', (10, 50), font, 3, (255, 255, 255), 5, cv2.LINE_AA)

# Display the image
cv2.imshow('Image with text', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

### Now put text inside the video or webcam 
import cv2

# Create a cv2.VideoCapture object
cap = cv2.VideoCapture("zia.mp4")

# Check if the video was opened successfully
if not cap.isOpened():
    print("Error opening video file")

# Read and process each frame of the video
while cap.isOpened():
    # Read the next frame , return two things 1. frame, 2. boolean value(True or False)---> True when frame is 
    # successfully read, False when frame is not successfully read
    ret, frame = cap.read()

    # Check if we reached the end of the video
    if not ret:
        break

    # Define the font to use
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Draw text on the frame
    cv2.putText(frame, 'Hello, World!', (10, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow("Frame", frame)

    # Wait for a key press
    key = cv2.waitKey(1)

    # if 'q' pressed then break the loop exit from it
    if key == ord("q"):
        break

# Release the video stream and close all windows
cap.release()
cv2.destroyAllWindows()

### what is cv2.minAreaRect(), and how it can used?

cv2.minAreaRect() is a function in the OpenCV (Open Computer Vision) library that can be used to find the minimum area rectangle that encloses a set of points. The function returns a rotated rectangle, which is defined by the center point, size, and orientation of the rectangle.
import cv2
import numpy as np

# Define a set of points
points = np.array([[10, 10], [100, 10], [100, 100], [10, 100]], np.int32)

# Find the minimum area rectangle that encloses the points
rect = cv2.minAreaRect(points)
print(rect)

# Extract the center, size, and orientation of the rectangle
(center_x, center_y), (width, height), angle = rect

# Draw the rectangle on an image
image = np.zeros((200, 200, 3), np.uint8)
box = cv2.boxPoints(rect)
print(box)
box = np.int0(box)
cv2.drawContours(image, [box], 0, (0, 0, 255), 2)

# Display the image
cv2.imshow('Image with rectangle', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

In this example, points is a 2D NumPy array containing the coordinates of the points that define the region we want to enclose with a rectangle. The cv2.minAreaRect() function takes this array as an input and returns the minimum area rectangle that encloses the points.

The cv2.boxPoints() function is used to convert the rotated rectangle returned by cv2.minAreaRect() into a list of points that can be used to draw the rectangle on an image using cv2.drawContours().
### Classes in Python

Python is an object oriented programming language.

Almost everything in Python is an object, with its properties and methods.

A Class is like an object constructor, or a "blueprint" for creating objects.

To create a class in Python, you use the class keyword followed by the name of the class and a colon. The class definition is then indented, and you can define various attributes and methods (functions) that belong to the class.
class name():

    def nameWin():
        print("zia")

    def abd():
        print("sweden")

    def juan():
        print("south")

name.juan()
name.abd()

obj = name

obj.juan()

### The __init__() Function

To understand the meaning of classes we have to understand the built-in __init__() function.

All classes have a function called __init__(), which is always executed when the class is being initiated.

Use the __init__() function to assign values to object properties, or other operations that are necessary to do when the object is being created:
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def printVal(self):
        print(self.name)
        print(self.age)



p1 = Person("John", 36)

p1.printVal()
In Python, the self keyword is used to refer to the instance of the object itself within the body of a class method. It is equivalent to the this keyword in other object-oriented languages.

Here is an example of how self is used in a class method in Python:
class MyClass:
    def __init__(self, value):
        self.value = value
    
    def print_value(self):
        print(self.value)

obj = MyClass(10)
obj.print_value()  # prints 10

### 1. Make a Function to Detect Contours Using Binary Thresholding 
import cv2 as cv

def detectUsingThresh(img, threshVal = 140):
        
        # convert to grayscale
        imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # invert grascale
        invertGray = cv.bitwise_not(imgGray)
        
        # apply binary thresholding
        ret, binaryThresh = cv.threshold(invertGray, threshVal, 255, cv.THRESH_BINARY)

        # detect contours 
        contours, hierarchy = cv.findContours(binaryThresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

        objectContours = []

        # save the contours with area atleast greater than 2000 pixels (50x40 = 2000)
        for cont in contours:
            area = cv.contourArea(cont)

            if area > 2000:
                objectContours.append(cont)

        return objectContours
import matplotlib.pyplot as plt

img = cv.imread("1.jpeg")
img = cv.resize(img, (400, 600))


contours = detectUsingThresh(img)
print(len(contours))

cv.drawContours(img, contours, -1, (0,0,255), 3)

plt.figure(figsize=(5,5))
plt.imshow(img)
plt.axis(False)
# cv.imshow("image", img)
# cv.waitKey()

cv.destroyAllWindows()
### 2. Make a Function to Detect Contours Using Canny filter
import cv2 as cv
import numpy as np

def detectUsingCanny(img, lower= 120,upper= 160):
        
        # Blur the image to remove noise
        blurred_image = cv.GaussianBlur(img,(5,5),0)

        # Apply canny edge detection
        cannyImge = cv.Canny(blurred_image, lower, upper)
        
        # kernel for image dilation
        dilateKernel = np.ones((5,5), "uint8")

        # image dilation to enhance the edges
        imgDilation = cv.dilate(cannyImge, dilateKernel, iterations=1)
        
        contours, hierarchy = cv.findContours(imgDilation, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        cv.imshow("dilat", imgDilation)

        objectContours = []

        for cont in contours:
            area = cv.contourArea(cont)

            if area > 2000:
                objectContours.append(cont)

        return objectContours
import matplotlib.pyplot as plt

img = cv.imread("1.jpeg")
img = cv.resize(img, (400, 600))


contours = detectUsingCanny(img)
print(len(contours))

cv.drawContours(img, contours, -1, (0,0,255), 3)

plt.figure(figsize=(5,5))
plt.imshow(img)
plt.axis(False)
# cv.imshow("image", img)
cv.waitKey()

cv.destroyAllWindows()
### Which one is better Binary Thresholding or Canny filter
Both binary thresholding and the Canny edge filter are commonly used techniques for detecting edges and contours in images. The choice of which method to use depends on the specific requirements of your application.

Binary thresholding is a simple and fast method for separating objects in an image from the background. It works by thresholding an image so that pixels with values above a certain threshold are set to one value (usually white), and pixels with values below the threshold are set to another value (usually black). This results in a binary image with white pixels representing object pixels and black pixels representing background pixels. Binary thresholding is suitable for images with clear and distinct object boundaries.

The Canny edge filter is a more sophisticated method for detecting edges and contours in images. It uses a multi-stage algorithm to detect edges that are more likely to correspond to object boundaries. The Canny edge filter works by first smoothing the image using a Gaussian filter to reduce noise, then finding the gradient intensity and direction of the image using the Sobel operator. It then applies non-maximum suppression to thin the edges and removes false edges caused by noise. Finally, it applies hysteresis thresholding to suppress weak edges and retain only strong edges. The Canny edge filter is more robust than binary thresholding and can handle images with more complex and varied edge features. However, it is also slower and more computationally intensive.

In general, the Canny edge filter is a better choice for contour detection in images with complex edge features or low contrast, while binary thresholding is a good choice for images with clear and distinct object boundaries. Ultimately, the best method for contour detection will depend on the characteristics of the images you are working with and the specific requirements of your application.
### Now make class with name MyDetectionMethods and include the above created function in it
import cv2 as cv
import numpy as np

class MyDetectionMethods():
    def __init__(self):
        pass

    def detectUsingThresh(self, img, threshVal = 140):
        
        # convert to grayscale
        imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # invert grascale
        invertGray = cv.bitwise_not(imgGray)
        
        # apply binary thresholding
        ret, binaryThresh = cv.threshold(invertGray, threshVal, 255, cv.THRESH_BINARY)

        # detect contours 
        contours, hierarchy = cv.findContours(binaryThresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

        objectContours = []

        # save the contours with area atleast greater than 2000 pixels (50x40 = 2000)
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
        
        # kernel for image dilation
        dilateKernel = np.ones((5,5), "uint8")

        # image dilation to enhance the edges
        imgDilation = cv.dilate(cannyImge, dilateKernel, iterations=1)
        
        contours, hierarchy = cv.findContours(imgDilation, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        cv.imshow("dilat", imgDilation)

        objectContours = []

        for cont in contours:
            area = cv.contourArea(cont)

            if area > 2000:
                objectContours.append(cont)

        return objectContours
import matplotlib.pyplot as plt

# make object of above created class
detector = MyDetectionMethods

img = cv.imread("1.jpeg")
img = cv.resize(img, (400, 600))

# now call the above function from class using object detector
contours = detector.detectUsingCanny('self', img=img, lower=120, upper=160)
print(len(contours))

cv.drawContours(img, contours, -1, (0,0,255), 3)

plt.figure(figsize=(5,5))
plt.imshow(img)
plt.axis(False)
# cv.imshow("image", img)
cv.waitKey()

cv.destroyAllWindows()

### Aruco Library of OpenCV

The Aruco library is a part of the open source computer vision library OpenCV (OpenCV stands for Open Source Computer Vision). It is a collection of tools and functions for detecting and identifying augmented reality (AR) markers in images and video streams. AR markers are small, square-shaped black and white patterns that are used to represent virtual objects in the real world.

The Aruco library provides a set of functions for detecting and identifying AR markers in images. It can detect markers of different sizes and types, and it can also estimate the pose (position and orientation) of the markers in the image. This information can be used to overlay virtual objects on top of the markers in real-time, creating the illusion of augmented reality.

To use the Aruco library in your Python code, you will need to install the OpenCV library and import the cv2.aruco module. Here is an example of how you can use the Aruco library to detect and identify AR markers in an image:
import cv2

# Load the image
image = cv2.imread("111.jpeg")

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Create a dictionary of AR markers
dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)

# Detect the markers in the image
corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary)

# If markers were detected, draw them on the image
if ids is not None:
    image = cv2.aruco.drawDetectedMarkers(image, corners, ids)

# Show the image
cv2.imshow("Image", image)
cv2.waitKey(0)

cv2.destroyAllWindows()

### Detect and draw AR markers in video or webcam
import cv2

# Load the image
image = cv2.imread("111.jpeg")

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Create a dictionary of AR markers
dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)

# Detect the markers in the image
corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary)

# If markers were detected, draw them on the image
if ids is not None:
    image = cv2.aruco.drawDetectedMarkers(image, corners, ids)

# Show the image
cv2.imshow("Image", image)
cv2.waitKey(0)

cv2.destroyAllWindows()

This code will detect and draw the boundaries of any AR markers present in the image. You can then use the ids and corners variables to identify the markers and estimate their pose.

The Aruco library is a powerful tool for creating augmented reality applications. It is widely used in a variety of fields, including robotics, industrial automation, and entertainment.
cv2.destroyAllWindows()
#### cv.aruco.DetectorParameters_create()

#### cv.aruco.Dictionary_get(cv.aruco.DICT_5X5_50)
