import cv2
import time

class FaceDetector:
    def __init__(self,
                 face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'),
                 eye_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')):

        self.face_cascade = face_cascade 
        self.eye_cascade = eye_cascade 

    def find_face(self,
                  frame,
                  draw=True,
                  face_color=(255, 0, 0),
                  eye_color=(0, 255, 0),
                  thickness=2):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:

            if draw:
                cv2.rectangle(frame, (x, y), (x + w, y + h), face_color, thickness)

            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.3, 5)

            for (ex, ey, ew, eh) in eyes:
                if draw:
                    cv2.rectangle(roi_color,
                                  (ex, ey),
                                  (ex + ew, ey + eh),
                                  eye_color,
                                  thickness)

        return frame


def main():
    cap = cv2.VideoCapture(0)
    detector = FaceDetector()
    ptime = 0

    while True:
        success, frame = cap.read()
        frame = cv2.flip(frame, 1)

        if not success:
            break

        frame = detector.find_face(frame)

 
        ctime = time.time()
        fps = 0 if ptime == 0 else 1 / (ctime - ptime)
        ptime = ctime

        cv2.putText(frame,
                    f"FPS: {int(fps)}",
                    (10, 70),
                    cv2.FONT_HERSHEY_PLAIN,
                    3,
                    (255, 0, 255),
                    3)

        cv2.imshow("Hand Module", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()