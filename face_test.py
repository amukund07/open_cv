import cv2
import time
import face_module as fm

cap = cv2.VideoCapture(0)
detector = fm.FaceDetector()

pTime = 0

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)

    if not success:
        break

    frame = detector.find_face(frame)

    cTime = time.time()
    fps = 0 if pTime == 0 else 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(frame,
                f"FPS: {int(fps)}",
                (10, 70),
                cv2.FONT_HERSHEY_PLAIN,
                3,
                (255, 0, 255),
                3)

    cv2.imshow("Face Module", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()