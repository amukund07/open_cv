import cv2
import numpy as np  
cap = cv2.VideoCapture(0)
cap= cv2.resize(cap,(0,0),fx=2,fy=2)
while True:
    ret, frame = cap.read()

    frame= cv2.resize(frame,(0,0),fx=2,fy=2)
    frame = cv2.flip(frame, 1)

    height, width = frame.shape[:2]   

    image = np.zeros(frame.shape, np.uint8)

    smaler_frame = cv2.resize(frame,(0,0),fx=0.5,fy=0.5)

    image[:height//2, :width//2] = smaler_frame          
    image[height//2:, :width//2] = smaler_frame          
    image[:height//2, width//2:] = smaler_frame          
    image[height//2:, width//2:] = smaler_frame          

    cv2.imshow('frame', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()