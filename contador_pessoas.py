import cv2 as cv 

cap = cv.VideoCapture('example_01.mp4')

if(cap.isOpened() == False):
    print("Deu ruim o video")

while(cap.isOpened()):
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    ret1,th = cv.threshold(gray,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    if ret == True:
        cv.imshow('tresh', th)
        if cv.waitKey(25) & 0xFF == ord('q'):
            break
    else: 
        break

cap.release()
cv.destroyAllWindows()