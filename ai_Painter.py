import numpy as np
import cv2
import hand_module as htm
import os
import time
#  HEADER IMAGES 
folder_path = "Header"
myList = os.listdir(folder_path)
overlayList = []

for imPath in myList:
    image = cv2.imread(f'{folder_path}/{imPath}')
    overlayList.append(image)

header = overlayList[0]

#  COLORS 
colour = [
    (0, 0, 255),     # Red
    (255, 55, 0),     # Blue
    (0, 255, 255),   # Yellow
    (0, 255, 0)      # Green
]

drawcolour = colour[0]
ptime=0
thickness = 8
eraserThickness = 100

#  VARIABLES 
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

#  CAMERA 
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.HandDetector()

#  MAIN LOOP 
while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)

    # Put header on top
    frame[0:125, 0:1280] = header

    # Detect hands
    frame, hands = detector.find_hands(frame, draw=False)

    if len(hands) != 0:
        lmList = hands[0]

        x1, y1 = lmList[8][1:]   # Index finger
        x2, y2 = lmList[12][1:]  # Middle finger

        fingers = detector.fingersUp(lmList)

        #  SELECTION MODE 
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0

            if y1 < 125:  # Only inside header

                if 214<x1<450:
                    header= overlayList[0]
                    drawcolour=colour[0]
                elif 450<x1<600:
                    header= overlayList[1]
                    drawcolour=colour[1]
                elif 600<x1<840:
                    header= overlayList[2]
                    drawcolour=colour[2]   
                elif 840<x1<1050:
                    header= overlayList[3]  
                    drawcolour=colour[3] 
                else:
                    header= overlayList[4]
                    drawcolour= (0,0,0)

            cv2.rectangle(frame, (x1, y1-25),
                          (x2, y2+25),
                          drawcolour, cv2.FILLED)

        #  DRAWING MODE 
        elif fingers[1] and not fingers[2]:

            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawcolour == (0, 0, 0):
                cv2.line(imgCanvas, (xp, yp),
                         (x1, y1),
                         drawcolour,
                         eraserThickness)
            else:
                cv2.line(imgCanvas, (xp, yp),
                         (x1, y1),
                         drawcolour,
                         thickness)

            xp, yp = x1, y1

        else:
            xp, yp = 0, 0

    #  MERGE CANVAS WITH FRAME 
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)

    frame = cv2.bitwise_and(frame, imgInv)
    frame = cv2.bitwise_or(frame, imgCanvas)
    
    ctime = time.time()
    fps = 1 / (ctime - ptime) if (ctime - ptime) != 0 else 0
    ptime = ctime

    cv2.putText(frame, f'FPS: {int(fps)}', (10, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (225, 0, 225), 3)

    # Show final output
    cv2.imshow("AI Virtual Painter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()








