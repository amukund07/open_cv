import cv2
import numpy as np  

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame= cv2.resize(frame,(0,0),fx=2,fy=2)
    frame = cv2.flip(frame, 1)
    hsv= cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    low_blue= np.array([0,51,51])
    hig_blue= np.array([224,255,255])
    mask= cv2.inRange(hsv,low_blue,hig_blue)
    result= cv2.bitwise_and(frame,frame,mask=mask)

    cv2.imshow('frame', result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()