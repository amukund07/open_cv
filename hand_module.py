import cv2
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import RunningMode


class HandDetector:
    def __init__(self,
                 model_path="hand_landmarker.task",
                 num_hands=2,
                 connections=None):

        self.model_path = model_path
        self.num_hands = num_hands

        base_options = python.BaseOptions(model_asset_path=model_path)

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=num_hands,
            running_mode=RunningMode.VIDEO
        )

        self.detector = vision.HandLandmarker.create_from_options(options)
        self.frame_timestamp = 0

        # Default connections if none provided
        self.default_connections = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (9,10),(10,11),(11,12),
            (13,14),(14,15),(15,16),
            (0,17),(17,18),(18,19),(19,20),
            (5,9),(9,13),(13,17)
        ]

        self.connections = connections if connections else self.default_connections



    def find_hands(self,
                   frame,
                   draw=True,
                   point_color=(0,255,0),
                   line_color=(255,0,0),
                   point_radius=5):

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        self.frame_timestamp += 1
        result = self.detector.detect_for_video(mp_image,
                                                self.frame_timestamp)

        h, w, _ = frame.shape
        all_hands = []

        if result.hand_landmarks:

            for hand_landmarks in result.hand_landmarks:

                lm_list = []

                for id, landmark in enumerate(hand_landmarks):
                    cx = int(landmark.x * w)
                    cy = int(landmark.y * h)

                    lm_list.append((id, cx, cy))

                    if draw:
                        cv2.circle(frame,
                                   (cx, cy),
                                   point_radius,
                                   point_color,
                                   -1)

                all_hands.append(lm_list)

                if draw:
                    for connection in self.connections:
                        start = hand_landmarks[connection[0]]
                        end = hand_landmarks[connection[1]]

                        x1 = int(start.x * w)
                        y1 = int(start.y * h)
                        x2 = int(end.x * w)
                        y2 = int(end.y * h)

                        cv2.line(frame,
                                 (x1, y1),
                                 (x2, y2),
                                 line_color,
                                 2)

        return frame, all_hands



    def find_position(self, frame, hand_no=0):
        frame, hands = self.find_hands(frame, draw=False)

        if len(hands) > hand_no:
            return hands[hand_no]
        return []



    def find_distance(self, hand, id1, id2):

        if len(hand) == 0:
            return 0

        x1, y1 = hand[id1][1], hand[id1][2]
        x2, y2 = hand[id2][1], hand[id2][2]

        length = math.hypot(x2 - x1, y2 - y1)

        return length



def main():
    import time

    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    ptime = 0

    while True:
        success, frame = cap.read()
        frame = cv2.flip(frame, 1)

        if not success:
            break

        frame, hands = detector.find_hands(frame)

 
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