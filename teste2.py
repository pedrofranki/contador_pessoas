import cv2
import numpy as np


def criaModelo(history, old, alpha):
    new = np.zeros(history[0].shape[:2]).astype('float32')
    for i in range(len(history)):
        new += history[i]

    new = new/len(history)


    return (1-alpha) * old + (alpha * new)




cap = cv2.VideoCapture('./videos/example_02.mp4')
ATTFRAMES = 50
history = []
alpha = 0.6
threshold = 0.2
modelo = np.zeros(1)
kernel = np.ones((7, 7), np.uint8)
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = gray.astype('float32')/255
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    if modelo.shape == (1,): #inicializando
        modelo = np.zeros(frame.shape[:2]).astype('float32')
    history.append(gray)
    if len(history) > ATTFRAMES:#atualizando modelo
        modelo = criaModelo(history, modelo, alpha)
        history = []


    cv2.imshow('modelo', modelo)
    diff = cv2.absdiff(modelo, gray)
    cv2.imshow('diferenca', diff)
    thresh = cv2.threshold(diff, threshold, 1, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    thresh = (thresh*255).astype('uint8')
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) < 1000:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('mapa', thresh)
    cv2.imshow('video', frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
