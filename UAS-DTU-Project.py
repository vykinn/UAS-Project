#importing OpenCV module
import cv2
#importing numpy module and abbrevating it as np
import numpy as np

#Loading the image
img = cv2.imread('sample_image.jpeg')

#Converting the image to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#Defining lower and upper bounds for brown and green regions
lower_brown = np.array([10, 50, 50])
upper_brown = np.array([20, 255, 255])
lower_green = np.array([40, 50, 50])
upper_green = np.array([70, 255, 255])

#Creating masks for both the regions
mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
mask_green = cv2.inRange(hsv, lower_green, upper_green)

#Find contours in the masks
contours_brown, _ = cv2.findContours(mask_brown, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours_green, _ = cv2.findContours(mask_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#Drawing contours on original image
img_contours = img.copy()
cv2.drawContours(img_contours, contours_brown, -1, (0, 0, 255), 2)
cv2.drawContours(img_contours, contours_green, -1, (0, 255, 0), 2)

# Loop through the contours and detect the color of each house
houses_brown = []
houses_green = []
for contour in contours_brown:
    x, y, w, h = cv2.boundingRect(contour)
    house_color = img[y + h // 2, x + w // 2]
    if np.array_equal(house_color, [255, 0, 0]):  # Blue house
        houses_brown.append(2)
    elif np.array_equal(house_color, [0, 0, 255]):  # Red house
        houses_brown.append(1)
for contour in contours_green:
    x, y, w, h = cv2.boundingRect(contour)
    house_color = img[y + h // 2, x + w // 2]
    if np.array_equal(house_color, [255, 0, 0]):  # Blue house
        houses_green.append(2)
    elif np.array_equal(house_color, [0, 0, 255]):  # Red house
        houses_green.append(1)

#Calculating total priority of houses on the brown and green regions
priority_brown = sum(houses_brown)
priority_green = sum(houses_green)

n_houses=np.array([houses_brown,houses_green])

#Displaying the output image 
cv2.imshow('Original Image', img)
cv2.imshow('Brown Region', mask_brown)
cv2.imshow('Green Region', mask_green)
cv2.imshow('Contours', img_contours)
cv2.waitKey(0)
cv2.destroyAllWindows()