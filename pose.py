import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import RunningMode



base_options = python.BaseOptions(
    model_asset_path="pose_landmarker_full.task"
)

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=RunningMode.VIDEO,
    num_poses=1
)

detector = vision.PoseLandmarker.create_from_options(options)


POSE_CONNECTIONS = [
    (11, 13), (13, 15),   # Left arm
    (12, 14), (14, 16),   # Right arm
    (11, 12),             # Shoulders
    (11, 23), (12, 24),   # Torso sides
    (23, 24),             # Hips
    (23, 25), (25, 27),   # Left leg
    (24, 26), (26, 28)    # Right leg
]

cap = cv2.VideoCapture(0)

ptime = 0

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    frame_timestamp = int(time.time() * 1000)

    result = detector.detect_for_video(mp_image, frame_timestamp)


    if result.pose_landmarks:
        for pose_landmarks in result.pose_landmarks:

          
            for landmark in pose_landmarks:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

       
            for connection in POSE_CONNECTIONS:
                start = pose_landmarks[connection[0]]
                end = pose_landmarks[connection[1]]

                x1 = int(start.x * w)
                y1 = int(start.y * h)
                x2 = int(end.x * w)
                y2 = int(end.y * h)

                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)



    ctime = time.time()
    fps = 1 / (ctime - ptime) if (ctime - ptime) != 0 else 0
    ptime = ctime

    cv2.putText(frame, f'FPS: {int(fps)}', (10, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (225, 0, 225), 3)



    cv2.imshow("Pose Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()