#!/usr/bin/env python
# coding: utf-8

import cv2
from centroid_tracker import CentroidTracker

cap = cv2.VideoCapture('high1.avi')
ct = CentroidTracker()

ret, frame1 = cap.read()
ret, frame2 = cap.read()

while cap.isOpened():
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 20,255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for contour in contours:
        (x,y,w,h) = cv2.boundingRect(contour)
        
        # 700 - min threshold for bees
        if cv2.contourArea(contour) < 700:
            continue
        rects.append(cv2.boundingRect(contour))
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0,255,0), 2)
    #cv2.drawContours(frame1, contours, -1, (0,255,0),2)
    
    objects = ct.update(rects)
    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(frame1, text, (centroid[0] + 700, centroid[1] + 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame1, (centroid[0] + 700, centroid[1] + 150), 4, (0, 255, 0), -1)
    
    cv2.imshow("feed", frame1)
    frame1 = frame2
    ret, frame2 = cap.read()
    
    
    if cv2.waitKey(40) == 27:
        break

cv2.destroyAllWindows()
cap.release()



