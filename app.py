import cv2
import numpy as np
import os
from Vibe import Vibe
from time import time
path = "./videos/example_01.mp4"
cap = cv2.VideoCapture(path)

vibe = Vibe()
ret,frame = cap.read()
begin = time()
frame = frame.astype(np.int32)
vibe.init(frame)
end = time()
writer = None
firstFrame = None

# print(end - begin)
i = 0 
while(cap.isOpened()):
    ret,frame = cap.read()
    frame = frame.astype(np.int32)
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = vibe.test_and_update(frame)
    if writer is None:
        (H, W) = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter("saida.avi", fourcc, 30, (W, H), True)
    if firstFrame is None:
        firstFrame = frame
        continue
        
    # print(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
    closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel) 
    cv2.imwrite('./output/mask%i.jpg'%i, mask)
    writer.write(mask)
    i+=1
    print(i)

cap.release()




