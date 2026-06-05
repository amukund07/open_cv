"""
tracker_app.py
==============
Combined real-time tracker:
    • Face Mesh  (478-point MediaPipe FaceLandmarker)
    • Hand       (21-point MediaPipe HandLandmarker)
    • Pose       (33-point MediaPipe PoseLandmarker)

Keyboard controls
-----------------
  F   – toggle Face Mesh
  H   – toggle Hand tracking
  P   – toggle Pose tracking
  D   – toggle landmark debug info panel
  Q   – quit

Requirements
------------
  pip install mediapipe opencv-python
  model files in the same directory:
    face_landmarker.task       (download from MediaPipe hub)
    hand_landmarker.task
    pose_landmarker_full.task
"""

import os
import sys
import time
import cv2

from facemesh_module import FaceMeshDetector
from hand_module    import HandDetector
from pose_module    import PoseDetector


# ── Paths ─────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
FACE_MODEL = os.path.join(BASE_DIR, "face_landmarker.task")
HAND_MODEL = os.path.join(BASE_DIR, "hand_landmarker.task")
POSE_MODEL = os.path.join(BASE_DIR, "pose_landmarker_full.task")


# ── Colour palette (BGR) ──────────────────────────────────────────────────
C_GREEN  = (0,   230, 120)
C_CYAN   = (0,   255, 220)
C_ORANGE = (0,   160, 255)
C_PINK   = (180,  60, 255)
C_WHITE  = (255, 255, 255)
C_DARK   = (20,   20,  20)
C_YELLOW = (0,   230, 255)
C_BLUE   = (255, 100,  30)


def draw_hud(frame, fps: float, toggles: dict, debug: bool,
             hands, faces, poses):
    """Overlay: FPS + module status + optional landmark counts."""
    h, w, _ = frame.shape

    # ── semi-transparent top bar ──
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), C_DARK, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # FPS
    cv2.putText(frame, f"FPS: {fps:5.1f}",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, C_YELLOW, 2)

    # Module toggles
    labels = [
        ("F-Mesh",  toggles["face"],  C_GREEN),
        ("Hands",   toggles["hand"],  C_CYAN),
        ("Pose",    toggles["pose"],  C_ORANGE),
    ]
    x_off = 200
    for name, active, col in labels:
        color = col if active else (80, 80, 80)
        cv2.putText(frame, f"[{'ON ' if active else 'OFF'}] {name}",
                    (x_off, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        x_off += 200

    # Key hints (bottom bar)
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - 30), (w, h), C_DARK, -1)
    cv2.addWeighted(overlay2, 0.55, frame, 0.45, 0, frame)
    hint = "  F: face mesh    H: hands    P: pose    D: debug    Q: quit"
    cv2.putText(frame, hint, (10, h - 8),
                cv2.FONT_HERSHEY_PLAIN, 1.2, (160, 160, 160), 1)

    # Debug panel
    if debug:
        lines = [
            f"Faces : {len(faces)} detected",
            f"Hands : {len(hands)} detected",
            f"Poses : {len(poses)} detected",
        ]
        if faces:
            ear_l = 0.0  # placeholder (computed externally)
            lines.append(f"Landmarks/face: {len(faces[0])}")
        if hands:
            lines.append(f"Landmarks/hand: {len(hands[0])}")

        panel_y = 60
        for ln in lines:
            cv2.putText(frame, ln, (w - 320, panel_y),
                        cv2.FONT_HERSHEY_PLAIN, 1.4, C_WHITE, 1)
            panel_y += 22

    return frame


def draw_face_info(frame, detector: FaceMeshDetector, faces, blendshapes):
    """Draw blink ratio & mouth-open status above each detected face."""
    h, w, _ = frame.shape
    for face_idx, face in enumerate(faces):
        # Face bounding box (rough)
        xs = [lm[1] for lm in face]
        ys = [lm[2] for lm in face]
        x1, y1 = min(xs), min(ys)

        ear_l = detector.blink_ratio(face, "left")
        ear_r = detector.blink_ratio(face, "right")
        open_ = detector.mouth_open(face)

        blink_color = C_PINK if (ear_l < 0.22 or ear_r < 0.22) else C_GREEN
        text = f"Blink L:{ear_l:.2f} R:{ear_r:.2f}  Mouth:{'OPEN' if open_ else 'shut'}"
        cv2.putText(frame, text,
                    (max(0, x1), max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_PLAIN, 1.3, blink_color, 1)

    return frame


def draw_hand_info(frame, hands):
    """Show finger-up count per hand."""
    tip_ids = [4, 8, 12, 16, 20]
    for hand_idx, hand in enumerate(hands):
        if len(hand) < 21:
            continue
        # Quick finger up count
        fingers = []
        # Thumb – compare x
        fingers.append(1 if hand[tip_ids[0]][1] > hand[tip_ids[0] - 1][1] else 0)
        for i in range(1, 5):
            fingers.append(1 if hand[tip_ids[i]][2] < hand[tip_ids[i] - 2][2] else 0)

        count = sum(fingers)
        cx, cy = hand[0][1], hand[0][2]   # wrist
        cv2.putText(frame, f"Hand {hand_idx + 1}: {count} finger(s)",
                    (cx - 40, cy + 30),
                    cv2.FONT_HERSHEY_PLAIN, 1.3, C_CYAN, 1)
    return frame


def draw_pose_info(frame, poses):
    """Draw shoulder-width and rough posture label."""
    for pose_idx, pose in enumerate(poses):
        if len(pose) < 25:
            continue
        # Shoulder width (id 11 & 12)
        l_sh = pose[11]
        r_sh = pose[12]
        sw = abs(l_sh[1] - r_sh[1])
        label = "Straight" if sw > 80 else "Turned"

        mid_x = (l_sh[1] + r_sh[1]) // 2
        mid_y = (l_sh[2] + r_sh[2]) // 2
        cv2.putText(frame, f"Pose: {label} | SW:{sw}px",
                    (mid_x - 60, mid_y - 20),
                    cv2.FONT_HERSHEY_PLAIN, 1.3, C_ORANGE, 1)
    return frame


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    # Check models
    missing = [m for m in [FACE_MODEL, HAND_MODEL, POSE_MODEL]
               if not os.path.exists(m)]
    if missing:
        print("⚠  Missing model file(s):")
        for m in missing:
            print(f"    {m}")
        print("\nDownload links:")
        print("  Face : https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task")
        print("  Hand : https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
        print("  Pose : https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task")
        sys.exit(1)

    # Initialise detectors
    print("Loading models …", flush=True)
    face_det = FaceMeshDetector(model_path=FACE_MODEL, num_faces=2)
    hand_det = HandDetector(model_path=HAND_MODEL, num_hands=2)
    pose_det = PoseDetector(model_path=POSE_MODEL)
    print("All models loaded. Starting camera …", flush=True)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # State
    toggles = {"face": True, "hand": True, "pose": True}
    debug   = False
    ptime   = 0.0

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)

        # Results holders
        faces, blendshapes = [], []
        hands              = []
        poses              = []

        # ── Run detectors ──────────────────────────────────────────────
        if toggles["face"]:
            frame, faces, blendshapes = face_det.find_face_mesh(
                frame,
                mesh_color=(0, 220, 110),
                oval_color=(0, 255, 220),
                iris_color=(30, 80, 255),
                point_radius=1,
                oval_thickness=2
            )
            frame = draw_face_info(frame, face_det, faces, blendshapes)

        if toggles["hand"]:
            frame, hands = hand_det.find_hands(
                frame,
                point_color=(0, 255, 180),
                line_color=(255, 120, 30),
                point_radius=5
            )
            frame = draw_hand_info(frame, hands)

        if toggles["pose"]:
            frame, poses = pose_det.find_pose(
                frame,
                point_color=(255, 80, 180),
                point_radius=5
            )
            frame = draw_pose_info(frame, poses)

        # ── FPS ────────────────────────────────────────────────────────
        ctime = time.time()
        fps   = 0.0 if ptime == 0 else 1.0 / max(ctime - ptime, 1e-6)
        ptime = ctime

        # ── HUD ────────────────────────────────────────────────────────
        frame = draw_hud(frame, fps, toggles, debug, hands, faces, poses)

        cv2.imshow("TensorCrew | Face Mesh + Hand + Pose Tracker", frame)

        # ── Key handling ───────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("f"):
            toggles["face"] = not toggles["face"]
        elif key == ord("h"):
            toggles["hand"] = not toggles["hand"]
        elif key == ord("p"):
            toggles["pose"] = not toggles["pose"]
        elif key == ord("d"):
            debug = not debug

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
