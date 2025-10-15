import cv2
import numpy as np

# MediaPipe pose edges (pairs of landmark indices to draw)
POSE_EDGES = [
    (11, 12), (11,13), (12,14), (13,15), (14,16),
    (23,24), (11,23), (12,24),
    (23,25), (24,26), (25,27), (26,28), (27,29), (28,30),
]

def draw_skeleton(frame, xy, color=(255,255,255), radius=3, thickness=2):
    f = frame.copy()
    for i,j in POSE_EDGES:
        p1 = tuple(np.int32(xy[i]))
        p2 = tuple(np.int32(xy[j]))
        cv2.line(f, p1, p2, color, thickness, cv2.LINE_AA)
    for p in xy:
        cv2.circle(f, tuple(np.int32(p)), radius, color, -1, cv2.LINE_AA)
    return f

class Trail:
    def __init__(self, maxlen=40):
        self.maxlen = maxlen
        self.pts = []

    def push(self, pt):
        self.pts.append(tuple(np.float32(pt)))
        if len(self.pts) > self.maxlen:
            self.pts.pop(0)

    def draw(self, frame, color=(200,200,200)):
        f = frame.copy()
        for i in range(1, len(self.pts)):
            p1 = tuple(np.int32(self.pts[i-1]))
            p2 = tuple(np.int32(self.pts[i]))
            cv2.line(f, p1, p2, color, 2, cv2.LINE_AA)
        return f

def overlay_bands_timeline(img_w, img_h, bands):
    """Create a small horizontal band image: green/amber/red blocks along time."""
    color_map = {'green': (60,180,75), 'amber': (255,225,25), 'red': (230,25,75)}
    bar = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    n = len(bands)
    for x in range(img_w):
        i = int(x * n / img_w)
        bar[:, x, :] = color_map[bands[i]]
    return bar
