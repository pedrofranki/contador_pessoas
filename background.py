import numpy as np 
import cv2 as cv 

cap = cv.VideoCapture('./videos/example_01.mp4')

ret, frame = cap.read()
cv.imwrite('background.jpg', frame)