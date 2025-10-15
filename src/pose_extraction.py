import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose

# List of landmark indices we will use often
POSE_LANDMARKS = mp_pose.PoseLandmark

def extract_keypoints_bgr_frame(frame_bgr, pose_model):
    """Return (landmarks_xy, visibility, landmarks_xyz) for a single BGR frame.
    landmarks_xy: (J,2), visibility: (J,), landmarks_xyz: (J,3) in normalized image coords if available.
    Returns None if not detected.
    """
    h, w = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = pose_model.process(frame_rgb)
    if not res.pose_landmarks:
        return None
    pts = res.pose_landmarks.landmark
    J = len(pts)
    xy = np.array([[p.x * w, p.y * h] for p in pts], dtype=np.float32)
    vis = np.array([p.visibility for p in pts], dtype=np.float32)
    xyz = np.array([[p.x, p.y, p.z] for p in pts], dtype=np.float32)
    return xy, vis, xyz

def iter_video_frames(path, max_frames=None):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield frame, count, fps
        count += 1
        if max_frames and count >= max_frames:
            break
    cap.release()
