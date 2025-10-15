import cv2
import numpy as np

def concat_side_by_side(a, b):
    h = max(a.shape[0], b.shape[0])
    def pad(img):
        if img.shape[0] == h: return img
        pad_top = (h - img.shape[0]) // 2
        pad_bot = h - img.shape[0] - pad_top
        return cv2.copyMakeBorder(img, pad_top, pad_bot, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
    a2, b2 = pad(a), pad(b)
    return np.hstack([a2, b2])

def resize_keep_ar(img, width=None, height=None):
    h, w = img.shape[:2]
    if width is None and height is None:
        return img
    if width is None:
        scale = height / h
    elif height is None:
        scale = width / w
    else:
        scale = min(width / w, height / h)
    new_w, new_h = int(w*scale), int(h*scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
