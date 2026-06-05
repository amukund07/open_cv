"""
facemesh_module.py
==================
Face Mesh detector using MediaPipe FaceLandmarker (Tasks API).
Follows the same module pattern as hand_module.py and pose_module.py.

Landmark groups (478 total points):
  - Face oval  : ~oval contour
  - Lips       : inner & outer
  - Left/Right eye + iris
  - Left/Right eyebrow
  - Nose

Usage:
    from facemesh_module import FaceMeshDetector
    detector = FaceMeshDetector()
    frame, faces, blendshapes = detector.find_face_mesh(frame)
"""

import cv2
import math
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import RunningMode


# ── Landmark index groups ──────────────────────────────────────────────────
FACE_OVAL_IDS = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361,
    288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149,
    150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
]

LEFT_EYE_IDS  = [33, 7, 163, 144, 145, 153, 154, 155, 133,
                  173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_IDS = [362, 382, 381, 380, 374, 373, 390, 249,
                  263, 466, 388, 387, 386, 385, 384, 398]
LEFT_IRIS_IDS  = [474, 475, 476, 477]
RIGHT_IRIS_IDS = [469, 470, 471, 472]

LEFT_EYEBROW_IDS  = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_EYEBROW_IDS = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]

NOSE_IDS = [1, 2, 98, 327, 168, 197, 195, 5]

LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
              185, 40, 39, 37, 0, 267, 269, 270, 409]
LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
              191, 80, 81, 82, 13, 312, 311, 310, 415]


class FaceMeshDetector:
    """
    MediaPipe FaceLandmarker wrapper.

    Parameters
    ----------
    model_path   : path to 'face_landmarker.task'
    num_faces    : max number of faces to detect
    min_detect   : minimum detection confidence
    min_track    : minimum tracking confidence
    draw_iris    : whether to draw iris landmarks
    running_mode : VIDEO (default) or IMAGE
    """

    def __init__(self,
                 model_path: str = "face_landmarker.task",
                 num_faces: int = 1,
                 min_detect: float = 0.5,
                 min_track: float = 0.5,
                 draw_iris: bool = True,
                 running_mode=RunningMode.VIDEO):

        self.draw_iris    = draw_iris
        self.running_mode = running_mode
        self.timestamp    = 0

        base_options = python.BaseOptions(model_asset_path=model_path)

        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode,
            num_faces=num_faces,
            min_face_detection_confidence=min_detect,
            min_face_presence_confidence=min_track,
            output_face_blendshapes=True          # expression scores
        )

        self.detector = vision.FaceLandmarker.create_from_options(options)

    # ── Core detection ─────────────────────────────────────────────────────
    def find_face_mesh(self,
                       frame,
                       draw: bool = True,
                       mesh_color: tuple = (0, 200, 100),
                       oval_color: tuple = (0, 255, 255),
                       iris_color: tuple = (0, 100, 255),
                       point_radius: int = 1,
                       oval_thickness: int = 2):
        """
        Detect and optionally draw face mesh on *frame*.

        Returns
        -------
        frame       : annotated frame (BGR)
        all_faces   : list of faces; each face is a list of (id, cx, cy, z)
        blendshapes : list of blendshape dicts per face  {name: score}
        """
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self.timestamp += 1

        if self.running_mode == RunningMode.VIDEO:
            result = self.detector.detect_for_video(mp_img, self.timestamp)
        else:
            result = self.detector.detect(mp_img)

        all_faces   = []
        blendshapes = []

        if not result.face_landmarks:
            return frame, all_faces, blendshapes

        for face_idx, face_lms in enumerate(result.face_landmarks):

            lm_list = []
            for lm_id, lm in enumerate(face_lms):
                cx = int(lm.x * w)
                cy = int(lm.y * h)
                lm_list.append((lm_id, cx, cy, lm.z))

                if draw:
                    cv2.circle(frame, (cx, cy), point_radius, mesh_color, -1)

            all_faces.append(lm_list)

            # Draw face oval
            if draw:
                self._draw_contour(frame, lm_list,
                                   FACE_OVAL_IDS, oval_color,
                                   oval_thickness, closed=True)

                self._draw_contour(frame, lm_list,
                                   LIPS_OUTER, (0, 0, 220), 1, closed=True)

                self._draw_contour(frame, lm_list,
                                   LIPS_INNER, (0, 0, 180), 1, closed=True)

                self._draw_contour(frame, lm_list,
                                   LEFT_EYE_IDS, (255, 200, 0), 1, closed=True)

                self._draw_contour(frame, lm_list,
                                   RIGHT_EYE_IDS, (255, 200, 0), 1, closed=True)

                if self.draw_iris:
                    self._draw_contour(frame, lm_list,
                                       LEFT_IRIS_IDS, iris_color, 1, closed=True)
                    self._draw_contour(frame, lm_list,
                                       RIGHT_IRIS_IDS, iris_color, 1, closed=True)

            # Blendshapes
            if result.face_blendshapes and face_idx < len(result.face_blendshapes):
                bs = {
                    c.category_name: round(c.score, 3)
                    for c in result.face_blendshapes[face_idx]
                }
                blendshapes.append(bs)
            else:
                blendshapes.append({})

        return frame, all_faces, blendshapes

    # ── Helpers ────────────────────────────────────────────────────────────
    def _draw_contour(self, frame, lm_list, ids, color, thickness, closed=False):
        """Draw a poly-line through landmark indices."""
        pts = [(lm_list[i][1], lm_list[i][2])
               for i in ids if i < len(lm_list)]
        if len(pts) < 2:
            return
        for k in range(len(pts) - 1):
            cv2.line(frame, pts[k], pts[k + 1], color, thickness)
        if closed:
            cv2.line(frame, pts[-1], pts[0], color, thickness)

    def get_landmark(self, face: list, lm_id: int):
        """Return (cx, cy, z) for a specific landmark index."""
        if lm_id < len(face):
            _, cx, cy, z = face[lm_id]
            return cx, cy, z
        return None

    def find_distance(self, face: list, id1: int, id2: int) -> float:
        """Euclidean pixel distance between two landmarks."""
        p1 = self.get_landmark(face, id1)
        p2 = self.get_landmark(face, id2)
        if p1 and p2:
            return math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        return 0.0

    def mouth_open(self, face: list, threshold: float = 25.0) -> bool:
        """Returns True if mouth appears open (upper-lower lip distance)."""
        dist = self.find_distance(face, 13, 14)   # upper & lower inner lip
        return dist > threshold

    def blink_ratio(self, face: list, eye: str = "left") -> float:
        """
        Eye aspect ratio (EAR) — smaller = more closed.
        eye: 'left' or 'right'
        """
        if eye == "left":
            h_dist = self.find_distance(face, 159, 145)  # vertical
            v_dist = self.find_distance(face, 33, 133)   # horizontal
        else:
            h_dist = self.find_distance(face, 386, 374)
            v_dist = self.find_distance(face, 362, 263)
        return round(h_dist / (v_dist + 1e-6), 3)

    def get_blendshape(self, blendshapes: dict, name: str) -> float:
        """Convenience getter for a blendshape score."""
        return blendshapes.get(name, 0.0)


# ── Stand-alone demo ───────────────────────────────────────────────────────
def main():
    import os
    model = "face_landmarker.task"
    if not os.path.exists(model):
        raise FileNotFoundError(
            f"Model not found: '{model}'\n"
            "Download from: https://storage.googleapis.com/mediapipe-models/"
            "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
        )

    cap      = cv2.VideoCapture(0)
    detector = FaceMeshDetector(model_path=model, num_faces=1)
    ptime    = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        frame, faces, blendshapes = detector.find_face_mesh(frame)

        # Show eye-blink ratio for first face
        if faces:
            ear_l = detector.blink_ratio(faces[0], "left")
            ear_r = detector.blink_ratio(faces[0], "right")
            open_ = detector.mouth_open(faces[0])

            cv2.putText(frame, f"EAR L:{ear_l:.2f} R:{ear_r:.2f}",
                        (10, 100), cv2.FONT_HERSHEY_PLAIN, 1.8, (200, 255, 50), 2)
            cv2.putText(frame, f"Mouth: {'OPEN' if open_ else 'closed'}",
                        (10, 130), cv2.FONT_HERSHEY_PLAIN, 1.8, (200, 255, 50), 2)

        ctime = time.time()
        fps   = 0 if ptime == 0 else 1 / (ctime - ptime)
        ptime = ctime

        cv2.putText(frame, f"FPS: {int(fps)}",
                    (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("Face Mesh Module", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
