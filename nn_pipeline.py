import os
import sys
import random
import time

import cv2
import numpy as np
import mss

from main import MONITOR, R, RING_MIN_S, RING_MAX_S, detect_rings

DATA = "dataset"
SPRITE_PATH = "approachcircle_2x.png"
CROP_TOP = 64                    # full-res rows above the playfield (HUD)
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


# ---------------------------------------------------------------------------
# Preprocessing (image processing step, shared by training and inference)
# ---------------------------------------------------------------------------
def preprocess(bgr):
    """Crop the HUD and normalize local contrast. Returns the cropped BGR image.

    Geometry is unchanged except for the crop, so labels only shift by CROP_TOP.
    """
    crop = bgr[CROP_TOP:]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = CLAHE.apply(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# ---------------------------------------------------------------------------
# Capture: record frames while you play
# ---------------------------------------------------------------------------
def capture(n=300, interval=0.5):
    os.makedirs(f"{DATA}/raw", exist_ok=True)
    sct = mss.MSS()
    for i in range(n):
        raw = np.asarray(sct.grab(MONITOR))[:, :, :3].copy()
        cv2.imwrite(f"{DATA}/raw/{i:04d}.png", raw)
        time.sleep(interval)
    print(f"Saved {n} frames to {DATA}/raw")


# ---------------------------------------------------------------------------
# Sprite compositing for synthetic rings
# ---------------------------------------------------------------------------
def load_sprite(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img.shape[2] == 3:
        bgr, alpha = img, np.full(img.shape[:2], 255, np.uint8)
    else:
        bgr, alpha = img[:, :, :3], img[:, :, 3]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    ys, xs = np.nonzero((gray >= 180) & (alpha >= 128))
    cx, cy = xs.mean(), ys.mean()
    sr = float(np.median(np.hypot(xs - cx, ys - cy)))
    return bgr, alpha, float(cx), float(cy), sr


SPR_BGR, SPR_A, SPR_CX, SPR_CY, SPR_R = load_sprite(SPRITE_PATH)


def paste_ring(frame, x, y, s, gain):
    """Alpha-composite the sprite so its ring is centered at (x, y) with radius s*R."""
    f = (s * R) / SPR_R
    nb = cv2.resize(SPR_BGR, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    na = cv2.resize(SPR_A, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    nh, nw = na.shape
    x0 = int(round(x - SPR_CX * f))
    y0 = int(round(y - SPR_CY * f))
    fx0, fy0 = max(0, x0), max(0, y0)
    fx1, fy1 = min(frame.shape[1], x0 + nw), min(frame.shape[0], y0 + nh)
    if fx1 <= fx0 or fy1 <= fy0:
        return
    sb = nb[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0].astype(np.float32)
    sa = (na[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0].astype(np.float32) / 255.0) * gain
    dst = frame[fy0:fy1, fx0:fx1].astype(np.float32)
    out = dst * (1 - sa[..., None]) + sb * sa[..., None]
    frame[fy0:fy1, fx0:fx1] = np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Label writing (YOLO format: class cx cy w h, normalized to the image)
# ---------------------------------------------------------------------------
def yolo_line(cx, cy, s, img_w, img_h):
    """Ring center in full-frame px. The box is the ring's diameter."""
    side = 2.0 * s * R
    cy_c = cy - CROP_TOP                      # shift into cropped coordinates
    return (f"0 {cx / img_w:.6f} {cy_c / img_h:.6f} "
            f"{side / img_w:.6f} {side / img_h:.6f}")


def build(synth_per_frame=4, val_frac=0.2, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    raws = sorted(os.listdir(f"{DATA}/raw"))
    items = []
    for name in raws:
        frame = cv2.imread(f"{DATA}/raw/{name}")
        gray_h = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (0, 0),
                            fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        dets, _ = detect_rings(gray_h)
        labels = [yolo_line(x, y, s, frame.shape[1], frame.shape[0] - CROP_TOP)
                  for x, y, s, _ in dets if s >= RING_MIN_S]
        items.append((name, frame, labels))

        # synthetic copies: paste extra rings onto the real frame
        for k in range(synth_per_frame):
            syn = frame.copy()
            syn_labels = list(labels)
            for _ in range(random.randint(1, 4)):
                s = random.uniform(RING_MIN_S, RING_MAX_S)
                x = random.uniform(s * R, frame.shape[1] - s * R)
                y = random.uniform(CROP_TOP + s * R, frame.shape[0] - s * R)
                paste_ring(syn, x, y, s, gain=random.uniform(0.4, 0.9))
                syn_labels.append(yolo_line(x, y, s, frame.shape[1],
                                            frame.shape[0] - CROP_TOP))
            items.append((f"{name[:-4]}_syn{k}.png", syn, syn_labels))

    random.shuffle(items)
    n_val = int(len(items) * val_frac)
    for split, subset in (("val", items[:n_val]), ("train", items[n_val:])):
        os.makedirs(f"{DATA}/images/{split}", exist_ok=True)
        os.makedirs(f"{DATA}/labels/{split}", exist_ok=True)
        for name, frame, labels in subset:
            stem = os.path.splitext(name)[0]
            img = preprocess(frame)
            cv2.imwrite(f"{DATA}/images/{split}/{stem}.png", img)
            with open(f"{DATA}/labels/{split}/{stem}.txt", "w") as fh:
                fh.write("\n".join(labels))

    with open(f"{DATA}/dataset.yaml", "w") as fh:
        fh.write(f"path: {os.path.abspath(DATA)}\n"
                 "train: images/train\nval: images/val\n"
                 "names:\n  0: approach\n")
    print(f"Built {len(items)} images ({n_val} val). Labels from detector + synthetic.")


# ---------------------------------------------------------------------------
# Training and inference
# ---------------------------------------------------------------------------
def train():
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    model.train(data=f"{DATA}/dataset.yaml", epochs=60, imgsz=640,
                batch=16, name="approach")


def load_model(weights="runs/detect/approach/weights/best.pt"):
    from ultralytics import YOLO
    return YOLO(weights)


def detect_nn(model, bgr_full, conf=0.35):
    """Return (x, y, s, conf) in full-frame px, same format as detect_rings."""
    img = preprocess(bgr_full)
    res = model.predict(img, conf=conf, imgsz=640, verbose=False)[0]
    dets = []
    for (x1, y1, x2, y2), c in zip(res.boxes.xyxy.cpu().numpy(),
                                    res.boxes.conf.cpu().numpy()):
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2 + CROP_TOP
        s = ((x2 - x1) / 2) / R
        if s >= RING_MIN_S:
            dets.append((float(cx), float(cy), float(s), float(c)))
    return dets


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "capture":
        capture(int(sys.argv[2]) if len(sys.argv) > 2 else 300)
    elif cmd == "build":
        build()
    elif cmd == "train":
        train()
    else:
        print("usage: python nn_pipeline.py [capture N | build | train]")