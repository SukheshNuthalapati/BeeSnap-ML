#!/usr/bin/env python
# coding: utf-8

import cv2
from centroid_tracker import CentroidTracker
import numpy as np
import random
from math import hypot
import matplotlib.pyplot as plt


cap = cv2.VideoCapture('high1.avi')
ct = CentroidTracker()

ret, frame1 = cap.read()
ret, frame2 = cap.read()
oldPoints = []
newPoints = []
colorsDict = {}
colors = []
pointsDict = {}


def updatePoints(objectID, centroid, oldCentroid):
    if objectID in pointsDict:
        dist = hypot(oldCentroid[1] - centroid[1], oldCentroid[0] - centroid[0])
        pointsDict[objectID].append(dist)
    else:
        pointsDict[objectID] = [0]

def getColor(objectID):
    if objectID in colorsDict:
        return colorsDict[objectID]
    else:
        r = random.randint(0,255)
        g = random.randint(0,255)
        b = random.randint(0,255)
        rgb = [r,g,b]
        colorsDict[objectID] = rgb
        return colorsDict[objectID]

while cap.isOpened():
    if frame1 is None or frame2 is None: 
        break

    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 20,255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rects = []

    for contour in contours:
        (x,y,w,h) = cv2.boundingRect(contour)
        
        if cv2.contourArea(contour) < 700:
            continue

        rects.append(cv2.boundingRect(contour))
        box = cv2.rectangle(frame1, (x,y),(x+ w,y+h),(0,255,0),2)
        
    for i in range(len(oldPoints)):
        oldcentroid = oldPoints[i]
        newcentroid = newPoints[i]
        cv2.line(frame1, (oldcentroid[0] + 700, oldcentroid[1] + 150), (newcentroid[0] + 700, newcentroid[1] + 150), colors[i], 2)
    objects, parents = ct.update(rects)

    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(frame1, text, (centroid[0] + 700, centroid[1] + 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, getColor(objectID), 2)
        oldCentroid = parents[objectID]
        colors.append(getColor(objectID))
        oldPoints.append(oldCentroid)
        newPoints.append(centroid)
        updatePoints(objectID, centroid, oldCentroid)
#         cv2.line(frame1, (oldCentroid[0] + 700, oldCentroid[1] + 150), (centroid[0] + 700, centroid[1] + 150), (0, 0, 255), 2)
        cv2.circle(frame1, (centroid[0] + 700, centroid[1] + 150), 4, getColor(objectID), -1)

    cv2.imshow("feed", frame1)
    frame1 = frame2
    ret, frame2 = cap.read()
    
    if cv2.waitKey(1) == ord('q'):
        break
import csv
# idList = []
# for ids in pointsDict:
#     idList.append(ids)
# with open('distances.csv', 'w', newline='') as csvfile:
#     spamwriter = csv.writer(csvfile, delimiter=',')
#     spamwriter.writerow(idList)

for ids in pointsDict:
    temp = np.array(pointsDict[ids])
    temp = [ids] + temp
    with open('distances.csv', 'a+', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        spamwriter.writerow(temp)

points = {}
for id in pointsDict:
    temp = np.array(pointsDict[id])
    #temp = np.cumsum(temp)
    #plt.plot(temp)
    points[id] = np.cumsum(temp)
   
last_points = []
for id in pointsDict:
    last_points.append(points.get(id)[-1])
    plt.scatter(id, points.get(id)[-1])
avg = np.mean(last_points)
plt.title("Total Distance For Each Bee" + " | Mean Distance:" + str(avg))
plt.show()

cv2.destroyAllWindows()
cap.release()