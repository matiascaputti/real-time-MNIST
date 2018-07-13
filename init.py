import cv2
import imutils
import numpy as np
from matplotlib import pyplot as plt

# get the reference to the webcam
CAMERA = cv2.VideoCapture(0)

CAPTURE_WIDTH = 700
ROI_LONG = 200 # Region Of Interest
MARGIN = 50
TOP = MARGIN
RIGHT = CAPTURE_WIDTH - MARGIN
BOTTOM = TOP + ROI_LONG
LEFT = RIGHT - ROI_LONG

while(True):
    _, frame = CAMERA.read()
    frame = imutils.resize(frame, CAPTURE_WIDTH)
    frame = cv2.flip(frame, 1)
    (height, width) = frame.shape[:2]

    # Add rectable to original frame
    cv2.rectangle(frame, (LEFT, TOP), (RIGHT, BOTTOM), (0,255,0), 2)
    cv2.imshow("Frame", frame)

    # Cut ROI
    roi = frame[TOP+2:BOTTOM-2, LEFT+2:RIGHT-2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    cv2.imshow("ROI", gray)

    # if the user pressed "q", then stop looping
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()