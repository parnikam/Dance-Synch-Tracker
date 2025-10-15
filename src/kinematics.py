import numpy as np

# Simple joint index helper map for readability (MediaPipe indices)
# See: https://google.github.io/mediapipe/solutions/pose#pose-landmark-model-blazePose-33-keypoints
L = {
    "L_SHOULDER": 11, "R_SHOULDER": 12,
    "L_ELBOW": 13, "R_ELBOW": 14,
    "L_WRIST": 15, "R_WRIST": 16,
    "L_HIP": 23, "R_HIP": 24,
    "L_KNEE": 25, "R_KNEE": 26,
    "L_ANKLE": 27, "R_ANKLE": 28,
    "NOSE": 0,
}

def angle(pA, pB, pC, eps=1e-8):
    BA = pA - pB
    BC = pC - pB
    num = (BA * BC).sum()
    den = np.linalg.norm(BA) * np.linalg.norm(BC) + eps
    c = np.clip(num / den, -1.0, 1.0)
    return np.degrees(np.arccos(c))

def compute_joint_angles(xy):
    """Compute a dict of key angles given a (J,2) array of keypoints in pixels."""
    ang = {}
    ang['L_ELBOW'] = angle(xy[L['L_SHOULDER']], xy[L['L_ELBOW']], xy[L['L_WRIST']])
    ang['R_ELBOW'] = angle(xy[L['R_SHOULDER']], xy[L['R_ELBOW']], xy[L['R_WRIST']])
    ang['L_KNEE']  = angle(xy[L['L_HIP']], xy[L['L_KNEE']], xy[L['L_ANKLE']])
    ang['R_KNEE']  = angle(xy[L['R_HIP']], xy[L['R_KNEE']], xy[L['R_ANKLE']])
    # Hip and shoulder flexion proxies
    ang['L_HIP']   = angle(xy[L['L_SHOULDER']], xy[L['L_HIP']], xy[L['L_KNEE']])
    ang['R_HIP']   = angle(xy[L['R_SHOULDER']], xy[L['R_HIP']], xy[L['R_KNEE']])
    ang['L_SHOULDER'] = angle(xy[L['L_ELBOW']], xy[L['L_SHOULDER']], xy[L['L_HIP']])
    ang['R_SHOULDER'] = angle(xy[L['R_ELBOW']], xy[L['R_SHOULDER']], xy[L['R_HIP']])
    return ang

def velocities(series, fps):
    """Finite-difference velocity and acceleration for a 1D array series."""
    v = np.gradient(series, 1.0/fps)
    a = np.gradient(v, 1.0/fps)
    return v, a

def center_of_mass(xy):
    """Simple CoM proxy: average of key torso + hip + knee + ankle points."""
    idx = [L['L_SHOULDER'], L['R_SHOULDER'], L['L_HIP'], L['R_HIP'], L['L_KNEE'], L['R_KNEE'], L['L_ANKLE'], L['R_ANKLE']]
    pts = xy[idx]
    return pts.mean(axis=0)

def jitter_bands(signal, fps, w_seconds=0.4, a_thresh=(150, 400)):
    """Return per-sample band labels 'green'|'amber'|'red' using accel magnitude windows.
    a_thresh: tuple(accel_for_amber, accel_for_red) in (deg/s^2 or px/s^2) depending on units.
    """
    import numpy as np
    w = max(1, int(w_seconds * fps))
    # accel magnitude via gradient of gradient
    a = np.gradient(np.gradient(signal, 1.0/fps), 1.0/fps)
    # moving average of |a| to reduce noise
    ma = np.convolve(np.abs(a), np.ones(w)/w, mode='same')
    amber_t, red_t = a_thresh
    bands = np.where(ma >= red_t, 'red', np.where(ma >= amber_t, 'amber', 'green'))
    return bands
