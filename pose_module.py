import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import RunningMode


class PoseDetector:
    def __init__(self,
                 model_path="pose_landmarker_full.task",
                 num_poses=1,
                 connections=None,
                 running_mode=RunningMode.VIDEO):

        self.model_path = model_path
        self.num_poses = num_poses
        self.running_mode = running_mode

        base_options = python.BaseOptions(
            model_asset_path=self.model_path
        )

        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=self.running_mode,
            num_poses=self.num_poses
        )

        self.detector = vision.PoseLandmarker.create_from_options(options)

        self.timestamp = 0

        self.default_connections = [
            (11, 13), (13, 15),   # Left arm
            (12, 14), (14, 16),   # Right arm
            (11, 12),             # Shoulders
            (11, 23), (12, 24),   # Torso sides
            (23, 24),             # Hips
            (23, 25), (25, 27),   # Left leg
            (24, 26), (26, 28)    # Right leg
        ]
        self.connections = connections if connections else self.default_connections


    def find_pose(self,
                  frame,
                  draw=True,
                  point_color=(0, 255, 0),
                  point_radius=5):

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        self.timestamp += 1

        result = self.detector.detect_for_video(
            mp_image,
            self.timestamp
        )

        h, w, _ = frame.shape
        all_poses = []

        if result.pose_landmarks:

            for pose_landmarks in result.pose_landmarks:

                lm_list = []

         
                for id, landmark in enumerate(pose_landmarks):
                    cx = int(landmark.x * w)
                    cy = int(landmark.y * h)

                    lm_list.append((id, cx, cy))

                    if draw:
                        cv2.circle(frame,
                                (cx, cy),
                                point_radius,
                                point_color,
                                -1)

   
                if draw:
                    for connection in self.connections:
                        id1, id2 = connection

                        x1, y1 = lm_list[id1][1], lm_list[id1][2]
                        x2, y2 = lm_list[id2][1], lm_list[id2][2]

                        cv2.line(frame,
                                (x1, y1),
                                (x2, y2),
                                (255, 0, 0),
                                2)

                all_poses.append(lm_list)
        return frame, all_poses
    

def main():

    cap = cv2.VideoCapture(0)
    detector = PoseDetector()

    ptime = 0

    while True:
        success, frame = cap.read()
        frame = cv2.flip(frame, 1)

        if not success:
            break

        frame, pose = detector.find_pose(frame)

 
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