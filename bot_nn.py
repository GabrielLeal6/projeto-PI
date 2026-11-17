import sys
import time
from collections import deque

import cv2
import keyboard
import mss
import numpy as np
from pynput.mouse import Controller

from main import MONITOR, R
from nn_pipeline import load_model, detect_nn

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
AR = 9.0                         # map's Approach Rate
MOUSE_LAG = 0.0                  # seconds the cursor should lead the hit
CONF_MIN = 0.35                  # network confidence threshold
ADMIT_MIN_S = 1.3                # only track rings still clearly outside the hit circle
GATE_D = 0.5                     # match gate: center distance, in units of R
GATE_S = 0.5                     # match gate: radius ratio difference
DEDUPE_D = 0.4                   # two detections this close (in R) and...
DEDUPE_S = 0.15                  # ...this close in size are the same ring
MIN_REMAIN = 0.06                # never aim as if the hit is closer than this (s)
MAX_DT = 0.05                    # clamp the frame step used for cursor motion
LOST_AFTER = 0.25                # seconds unseen before dropping a ring
GRACE = 0.10                     # seconds after hit time before dropping a ring
SWITCH_MARGIN = 0.15             # only switch targets if a new one is this much earlier
SNAP_KEY = "s"
DEBUG = True

PREEMPT = (1.2 - 0.15 * (AR - 5)) if AR >= 5 else (1.2 + 0.12 * (5 - AR))


def hit_time(s, now):
    """The ring shrinks from 4R to R over PREEMPT seconds.
    s is ring radius / R, so the ring is at R when s = 1."""
    return now + PREEMPT * max(s - 1.0, 0.0) / 3.0


class Track:
    def __init__(self, c, s, now):
        self.c = c
        self.s = s
        self.last_seen = now
        self.born = now
        self.hits = deque([hit_time(s, now)], maxlen=5)

    @property
    def hit(self):
        return float(np.median(self.hits))

    def update(self, c, s, now):
        self.c = c
        self.s = s
        self.last_seen = now
        self.hits.append(hit_time(s, now))


def same_ring(c1, s1, c2, s2):
    return (np.hypot(c1[0] - c2[0], c1[1] - c2[1]) < DEDUPE_D * R
            and abs(s1 - s2) < DEDUPE_S)


def save_debug(gray_or_bgr, dets, path="nn_snap.png"):
    """Draw detections at the correct radius (s * R)."""
    out = gray_or_bgr.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    for x, y, s, conf in dets:
        cv2.circle(out, (int(x), int(y)), int(round(s * R)), (0, 255, 0), 2)
        cv2.circle(out, (int(x), int(y)), int(round(R)), (0, 0, 255), 1)
        cv2.putText(out, f"{conf:.2f}", (int(x) + 4, int(y) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.imwrite(path, out)


def main():
    model = load_model()
    mouse = Controller()
    sct = mss.MSS()

    print(f"NN bot running. Hold '{SNAP_KEY}' to save nn_snap.png. Press 'q' to stop.")
    tracks = []
    current = None
    cursor = np.array(mouse.position, dtype=float)
    t_prev = time.perf_counter()
    last_dbg = 0.0
    admitted_window = 0
    mode = "idle"
    snap_held = False

    while not keyboard.is_pressed("q"):
        now = time.perf_counter()
        dt = min(now - t_prev, MAX_DT)
        frame_ms = (now - t_prev) * 1000

        raw = np.asarray(sct.grab(MONITOR))[:, :, :3].copy()
        t_det = time.perf_counter()
        dets = [d for d in detect_nn(model, raw, conf=CONF_MIN)]
        detect_ms = (time.perf_counter() - t_det) * 1000
        best_conf = max((d[3] for d in dets), default=0.0)

        if keyboard.is_pressed(SNAP_KEY):
            if not snap_held:
                save_debug(raw, dets)
                print(f"  saved nn_snap.png (rings={len(dets)})")
            snap_held = True
        else:
            snap_held = False

        # 1. Match detections to existing tracks
        taken = set()
        updated = 0
        for t in tracks:
            best_i, best_cost = None, None
            for i, (x, y, s, _) in enumerate(dets):
                if i in taken:
                    continue
                d = np.hypot(x - t.c[0], y - t.c[1])
                ds = abs(s - t.s)
                if d < GATE_D * R and ds < GATE_S:
                    cost = d / R + ds
                    if best_cost is None or cost < best_cost:
                        best_i, best_cost = i, cost
            if best_i is not None:
                taken.add(best_i)
                x, y, s, _ = dets[best_i]
                t.update((x, y), s, now)
                updated += 1

        # 2. Admit unmatched rings still outside the hit circle and not already tracked
        for i, (x, y, s, _) in enumerate(dets):
            if i in taken or s < ADMIT_MIN_S:
                continue
            if any(same_ring((x, y), s, t.c, t.s) for t in tracks):
                continue
            tracks.append(Track((x, y), s, now))
            admitted_window += 1

        # 3. Drop hit or lost rings, and merge duplicates
        kept = []
        for t in sorted(tracks, key=lambda t: t.born):
            if now - t.last_seen >= LOST_AFTER or now >= t.hit + GRACE:
                continue
            if any(same_ring(t.c, t.s, k.c, k.s) for k in kept):
                continue
            kept.append(t)
        tracks = kept
        if current is not None and current not in tracks:
            current = None

        # 4. Move toward the locked target
        cands = [t for t in tracks if t.hit >= now - 0.02]
        if cands:
            earliest = min(cands, key=lambda t: t.hit)
            if current is None or current not in cands or \
                    earliest.hit < current.hit - SWITCH_MARGIN:
                current = earliest
            t = current
            mode = "circle"
            target = np.array([MONITOR["left"] + t.c[0],
                               MONITOR["top"] + t.c[1]], dtype=float)
            remaining = max(t.hit - MOUSE_LAG - now, MIN_REMAIN)
            cursor += (target - cursor) * min(1.0, dt / remaining)
            mouse.position = (int(cursor[0]), int(cursor[1]))
        else:
            current = None
            mode = "idle"

        if DEBUG and now - last_dbg > 1.0:
            print(f"rings={len(dets):2d}  best={best_conf:.2f}  "
                  f"tracked={len(tracks):2d}  updated={updated:2d}  "
                  f"admitted={admitted_window:2d}  mode={mode}  "
                  f"detect={detect_ms:4.0f}ms  frame={frame_ms:4.0f}ms")
            last_dbg = now
            admitted_window = 0

        t_prev = now


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--snap":
        raw = np.asarray(mss.MSS().grab(MONITOR))[:, :, :3].copy()
        d = detect_nn(load_model(), raw, conf=CONF_MIN)
        save_debug(raw, d)
        print(f"Detected {len(d)} rings. Saved nn_snap.png.")
    else:
        main()