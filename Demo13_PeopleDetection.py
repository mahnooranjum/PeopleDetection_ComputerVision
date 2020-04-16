#==============================================================================
#   Tutorial By: Mahnoor Anjum
#   Date: 8/08/2018
#   Codes inspired by:
#   Rajeev Ratab
#   Official Documentation
#==============================================================================
import numpy as np
import cv2
# Classifier (XML file format)
people = cv2.CascadeClassifier('haarcascades/haarcascade_fullbody.xml')
#people = cv2.CascadeClassifier('haarcascades/haarcascade_lowerbody.xml')
#people = cv2.CascadeClassifier('haarcascades/haarcascade_upperbody.xml')
def detector(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    obj = people.detectMultiScale(gray)
    if obj is ():
        return image
    for (x,y,w,h) in obj:
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,255),2)
    return image

cap = cv2.VideoCapture('imgs/demo13.mp4')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('Demo13.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

while True:
    ret, frame = cap.read()
    cv2.imshow('People Detector', detector(frame))
    out.write(detector(frame))
    if cv2.waitKey(1)==13:
        break
out.release()
cap.release()
cv2.destroyAllWindows()
