import cv2 as cv 
import imutils

cap = cv.VideoCapture('./videos/example_01.mp4')

if(cap.isOpened() == False):
    print("Deu ruim o video")

firstFrame = None
W = None
H = None
writer = None

while(cap.isOpened()):
    ret, frame = cap.read()

    if ret == True:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray = cv.GaussianBlur(gray, (21, 21), 0)
        if writer is None:
            (H, W) = frame.shape[:2]
            fourcc = cv.VideoWriter_fourcc(*"MJPG")
            writer = cv.VideoWriter("saida.avi", fourcc, 30, (W, H), True)
        if firstFrame is None:
            firstFrame = gray
            continue
        
        frameDelta = cv.absdiff(firstFrame, gray)
        thresh = cv.threshold(frameDelta, 20, 255, cv.THRESH_BINARY)[1]
        #cv.imshow('thresh', thresh)
        thresh = cv.dilate(thresh, None, iterations=2)
        contours, hierarchy = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv.contourArea(c) < 1000:
                continue
            (x, y, w, h) = cv.boundingRect(c)
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


        writer.write(frame)
        #cv.imshow('Video', frame)
        if cv.waitKey(25) & 0xFF == ord('q'):
            break
    else: 
        break

cap.release()
cv.destroyAllWindows()