from nn_pipeline import load_model, detect_nn
import cv2, mss, numpy as np
from main import MONITOR
model = load_model()
raw = np.asarray(mss.MSS().grab(MONITOR))[:, :, :3].copy()
for x, y, s, c in detect_nn(model, raw):
    cv2.circle(raw, (int(x), int(y)), int(s * 54.4 * 2), (0, 255, 0), 2)
cv2.imwrite("nn_snap.png", raw)