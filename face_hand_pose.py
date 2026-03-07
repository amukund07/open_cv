import cv2
import time

from hand_module import HandDetector
from face_module import FaceDetector
from pose_module import PoseDetector

cap = cv2.VideoCapture(0)

hand_detector = HandDetector()
face_detector = FaceDetector()
pose_detector = PoseDetector()

pTime = 0

while True:
    success, frame = cap.read()
    frame= cv2.resize(frame,(0,0),fx=2,fy=2)
    frame = cv2.flip(frame, 1)

    if not success:
        break


    frame, hands = hand_detector.find_hands(frame)
    frame = face_detector.find_face(frame)
    frame, poses = pose_detector.find_pose(frame)

    # ---- FPS ----
    cTime = time.time()
    fps = 0 if pTime == 0 else 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(frame,
                f"FPS: {int(fps)}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2)

    cv2.imshow("Full CV System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()