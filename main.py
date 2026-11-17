import sys
import time
from collections import deque

import cv2
import keyboard
import mss
import numpy as np
from pynput.mouse import Controller

# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
MONITOR = {"top": 20, "left": 420, "width": 1130, "height": 990}
PLAYFIELD_Y = (64, 920)          # full-res rows to search (skips HUD)
AR = 9.0                         # map's Approach Rate
CS = 4.0                         # map's Circle Size
PX_PER_UNIT = 2.0                # full-res px per osu! unit
MOUSE_LAG = 0.0                  # seconds the cursor should lead the hit

RING_MIN_S = 1.3                 # approach ring radius / hit radius, lower bound
RING_MAX_S = 4.3                 # upper bound (rings start at about 4R)
HOUGH_EDGE = 80                  # Canny high threshold inside HoughCircles
HOUGH_VOTES = 35                 # accumulator threshold. Lower = more detections.
COVER_MIN = 0.50                 # fraction of ring circumference on edges
ADMIT_MIN_S = RING_MIN_S         # admit rings still clearly outside the hit radius
LOST_AFTER = 0.25                # seconds unseen before dropping a ring
GRACE = 0.10                     # seconds after hit time before dropping a ring
SWITCH_MARGIN = 0.15             # only switch targets if a new one is this much earlier
SNAP_KEY = "s"                   # hold while playing to save snap.png
DEBUG = True
DEBUG_DROPS = False

# ---------------------------------------------------------------------------
# Derived geometry
# ---------------------------------------------------------------------------
R = (54.4 - 4.48 * CS) * PX_PER_UNIT            # hit radius, full-res px
R_H = R / 2                                      # hit radius, half-res px
PREEMPT = (1.2 - 0.15 * (AR - 5)) if AR >= 5 else (1.2 + 0.12 * (5 - AR))
MIN_R_H = int(RING_MIN_S * R_H)                  # Hough radius range, half-res
MAX_R_H = int(RING_MAX_S * R_H)
MIN_DIST = max(4, int(R_H * 0.6))                # minimum center separation
ANGLES = np.linspace(0, 2 * np.pi, 72, endpoint=False)


def ring_coverage(edges, cx, cy, r):
    """Fraction of sample angles on the circle that land on an edge pixel."""
    h, w = edges.shape
    hit = np.zeros(ANGLES.shape, dtype=bool)
    for d in (-1.0, 0.0, 1.0):
        xs = np.clip(np.round(cx + (r + d) * np.cos(ANGLES)).astype(int), 0, w - 1)
        ys = np.clip(np.round(cy + (r + d) * np.sin(ANGLES)).astype(int), 0, h - 1)
        hit |= edges[ys, xs] > 0
    return float(hit.mean())


def detect_rings(gray_h):
    """Return (dets, best).

    dets: list of (x, y, s, coverage) in full-res px, where s = ring radius / R.
    best: (coverage, (x, y), s) for the strongest candidate, passed or not.
    """
    y0, y1 = PLAYFIELD_Y[0] // 2, PLAYFIELD_Y[1] // 2
    band = cv2.GaussianBlur(gray_h[y0:y1], (3, 3), 0)
    edges = cv2.Canny(band, HOUGH_EDGE // 2, HOUGH_EDGE)

    circles = cv2.HoughCircles(
        band, cv2.HOUGH_GRADIENT, dp=1, minDist=MIN_DIST,
        param1=HOUGH_EDGE, param2=HOUGH_VOTES,
        minRadius=MIN_R_H, maxRadius=MAX_R_H)

    dets = []
    best = (0.0, None, None)
    if circles is not None:
        for cx, cy, r in circles[0]:
            cov = ring_coverage(edges, cx, cy, r)
            fx, fy = 2.0 * cx, 2.0 * (cy + y0)
            s = (2.0 * r) / R
            if cov > best[0]:
                best = (cov, (fx, fy), s)
            if cov >= COVER_MIN:
                dets.append((fx, fy, s, cov))
    return dets, best


def hit_from_radius(r, now):
    """The ring shrinks from 4R to R over PREEMPT seconds."""
    return now + PREEMPT * max(r - R, 0.0) / (3 * R)


class Ring:
    def __init__(self, c, now, hit):
        self.c = c
        self.last_seen = now
        self.born = now
        self.samples = deque([hit], maxlen=5)
        self.admit_lead = hit - now

    @property
    def hit(self):
        return float(np.median(self.samples))


def drop_reason(t, now):
    if now - t.last_seen >= LOST_AFTER:
        return "lost"
    if now >= t.hit + GRACE:
        return "hit"
    return None


def save_debug(gray_full, dets, best, path="snap.png"):
    """Full frame with detections (green), hit radius (red), best (yellow)."""
    out = cv2.cvtColor(gray_full, cv2.COLOR_GRAY2BGR)
    for x, y, s, cov in dets:
        cv2.circle(out, (int(x), int(y)), int(round(s * R)), (0, 255, 0), 2)
        cv2.circle(out, (int(x), int(y)), int(round(R)), (0, 0, 255), 1)
    if best[1] is not None:
        bx, by = best[1]
        cv2.circle(out, (int(bx), int(by)), int(round(best[2] * R)), (0, 255, 255), 2)
        cv2.putText(out, f"best {best[0]:.2f}", (int(bx) + 6, int(by) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.imwrite(path, out)


def main():
    mouse = Controller()
    sct = mss.MSS()

    print(f"Bot running. Hold '{SNAP_KEY}' to save snap.png. Press 'q' to stop.")
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
        frame_ms = (now - t_prev) * 1000

        raw = np.asarray(sct.grab(MONITOR))
        gray = cv2.cvtColor(raw, cv2.COLOR_BGRA2GRAY)
        gray_h = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5,
                            interpolation=cv2.INTER_AREA)
        t_det = time.perf_counter()
        dets, best = detect_rings(gray_h)
        detect_ms = (time.perf_counter() - t_det) * 1000

        if keyboard.is_pressed(SNAP_KEY):
            if not snap_held:
                save_debug(gray, dets, best)
                print(f"  saved snap.png (rings={len(dets)}, best={best[0]:.2f})")
            snap_held = True
        else:
            snap_held = False

        # 1. Match detections to tracked rings by center
        taken = set()
        updated = 0
        for t in tracks:
            best_i, best_d = None, 0.5 * R
            for i, (x, y, s, _) in enumerate(dets):
                if i in taken:
                    continue
                d = np.hypot(x - t.c[0], y - t.c[1])
                if d < best_d:
                    best_i, best_d = i, d
            if best_i is not None:
                taken.add(best_i)
                x, y, s, _ = dets[best_i]
                t.c = (x, y)
                t.last_seen = now
                t.samples.append(hit_from_radius(s * R, now))
                updated += 1

        # 2. Admit new rings still clearly outside the hit radius
        for i, (x, y, s, _) in enumerate(dets):
            if i in taken or s < ADMIT_MIN_S:
                continue
            tracks.append(Ring((x, y), now, hit_from_radius(s * R, now)))
            admitted_window += 1

        # 3. Drop rings that were hit or lost
        kept = []
        for t in tracks:
            why = drop_reason(t, now)
            if why is None:
                kept.append(t)
            elif DEBUG_DROPS:
                print(f"  drop {why}: admit_lead={t.admit_lead:+.3f}s "
                      f"age={now - t.born:.2f}s at={tuple(int(v) for v in t.c)}")
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
            remaining = max(t.hit - MOUSE_LAG - now, 1e-3)
            dt = now - t_prev
            cursor += (target - cursor) * min(1.0, dt / remaining)
            mouse.position = (int(cursor[0]), int(cursor[1]))
        else:
            current = None
            mode = "idle"

        if DEBUG and now - last_dbg > 1.0:
            print(f"rings={len(dets):2d}  best={best[0]:.2f}  "
                  f"tracked={len(tracks):2d}  updated={updated:2d}  "
                  f"admitted={admitted_window:2d}  mode={mode}  "
                  f"detect={detect_ms:4.0f}ms  frame={frame_ms:4.0f}ms")
            last_dbg = now
            admitted_window = 0

        t_prev = now


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--snap":
        raw = np.asarray(mss.MSS().grab(MONITOR))
        g = cv2.cvtColor(raw, cv2.COLOR_BGRA2GRAY)
        gh = cv2.resize(g, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        d, b = detect_rings(gh)
        save_debug(g, d, b)
        print(f"Detected {len(d)} rings. Best coverage {b[0]:.2f}. Saved snap.png.")
    else:
        main()