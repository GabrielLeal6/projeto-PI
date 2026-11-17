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
PLAYFIELD_Y = (64, 920)        # full-res rows to search (skips HUD)
AR = 9.0                       # map's Approach Rate
CS = 4.0                       # map's Circle Size
PX_PER_UNIT = 2.0              # full-res px per osu! unit. Calibrate with --snap.
MOUSE_LAG = 0.0                # seconds the cursor should lead the hit
DETECT_THRESH = 0.5            # template match threshold (half-res)
SCALES = (0.9, 1.0, 1.1)       # template scales to tolerate a wrong PX_PER_UNIT
RING_LO = 1.2                  # ring search range, in units of R
RING_HI = 4.3
RING_MIN_PEAK = 4.0            # minimum absolute radial gradient for a ring
RING_PEAK_RATIO = 1.8          # ring peak must be this many times the median
OTHER_BLOCK = 1.15             # ignore samples within this * R of other circles
ADMIT_MIN_R = 1.3              # admit circles whose ring is still outside R
LOST_AFTER = 0.40              # seconds without a match before dropping
GRACE = 0.10                   # seconds after hit time before dropping
SWITCH_MARGIN = 0.15           # only switch targets if new one is this much earlier
ENABLE_SLIDERS = False         # slider handling is unverified. Keep off for now.
SLIDER_PX_PER_S = 900.0        # ball speed in full-res px/s (only if sliders on)
SLIDER_GAIN = 25.0             # cursor follow rate on sliders (only if sliders on)
BODY_LO, BODY_HI = 20, 110     # gray range of slider body fill
BODY_MIN_EXT = 2.5             # body must reach this many R beyond the head
BODY_MIN_TOUCH = 5             # body pixels that must touch the head
DEBUG = True

# ---------------------------------------------------------------------------
# Derived geometry (full-res pixels)
# ---------------------------------------------------------------------------
R = (54.4 - 4.48 * CS) * PX_PER_UNIT
PREEMPT = (1.2 - 0.15 * (AR - 5)) if AR >= 5 else (1.2 + 0.12 * (5 - AR))
BODY_OPEN = np.ones((3, 3), dtype=np.uint8)


def make_template(radius_half):
    """Circle outline template at half-res."""
    half = int(np.ceil(radius_half * 1.15)) + 2
    size = 2 * half + 1
    tpl = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(tpl, (half, half), int(round(radius_half)), 255, 2)
    return tpl, half


TEMPLATES = [make_template(R / 2 * s) for s in SCALES]
NMS_KERNEL = np.ones((int(R / 2) | 1,) * 2, dtype=np.uint8)


def detect_circles(gray_half):
    """Hit circle centers as (x, y) in full-res coordinates."""
    y0, y1 = PLAYFIELD_Y[0] // 2, PLAYFIELD_Y[1] // 2
    band = gray_half[y0:y1]
    peaks = []
    for tpl, half in TEMPLATES:
        if band.shape[0] <= tpl.shape[0] or band.shape[1] <= tpl.shape[1]:
            continue
        res = cv2.matchTemplate(band, tpl, cv2.TM_CCOEFF_NORMED)
        local_max = cv2.dilate(res, NMS_KERNEL)
        ys, xs = np.nonzero((res == local_max) & (res >= DETECT_THRESH))
        for x, y in zip(xs, ys):
            peaks.append((float(res[y, x]), int(x) + half, int(y) + half + y0))

    peaks.sort(reverse=True)
    kept = []
    min_sep = 0.6 * R / 2
    for _, cx, cy in peaks:
        if all(np.hypot(cx - px, cy - py) > min_sep for px, py in kept):
            kept.append((cx, cy))
    return [(2 * x, 2 * y) for x, y in kept]


def ring_profile(gray, cx, cy, others, r_lo, r_hi):
    """Median radial gradient in [r_lo, r_hi), ignoring samples inside other circles."""
    rs = np.arange(r_lo, r_hi, 1.0, dtype=np.float32)
    ang = np.linspace(0, 2 * np.pi, 48, endpoint=False, dtype=np.float32)
    mx = (cx + np.outer(np.cos(ang), rs)).astype(np.float32)
    my = (cy + np.outer(np.sin(ang), rs)).astype(np.float32)
    samp = cv2.remap(gray, mx, my, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE).astype(np.int16)

    bad = np.zeros(mx.shape, dtype=bool)
    for ox, oy in others:
        bad |= np.hypot(mx - ox, my - oy) < OTHER_BLOCK * R

    grad = np.abs(np.diff(samp, axis=1)).astype(np.float32)
    grad[bad[:, 1:] | bad[:, :-1]] = np.nan
    with np.errstate(all="ignore"):
        score = np.nanmedian(grad, axis=0)
    score = np.nan_to_num(score, nan=0.0)
    return rs[:-1] + 0.5, score


def find_ring(gray, cx, cy, others, pred=None):
    """Radius of the approach ring, or None. Searches only near pred when given."""
    lo, hi = RING_LO * R, RING_HI * R
    if pred is not None:
        lo = max(lo, pred - 0.5 * R)
        hi = min(hi, pred + 0.5 * R)
    if hi - lo < 4:
        return None
    rs, score = ring_profile(gray, cx, cy, others, lo, hi)
    i = int(np.argmax(score))
    peak = float(score[i])
    base = float(np.median(score)) + 1.0
    if peak < RING_MIN_PEAK or peak < RING_PEAK_RATIO * base:
        return None
    return float(rs[i])


def hit_from_radius(r, now):
    """Hit time from the ring radius. The ring shrinks 4R -> R over PREEMPT."""
    return now + PREEMPT * max(r - R, 0.0) / (3 * R)


def predicted_radius(hit, now):
    return R + 3 * R * max(hit - now, 0.0) / PREEMPT


def slider_body(gray_full, head, others):
    """Body pixels (x, y) beyond the head, or None if this is not a slider."""
    hx, hy = head
    pad = int(6 * R)
    h, w = gray_full.shape
    x0, x1 = max(0, hx - pad), min(w, hx + pad)
    y0, y1 = max(0, hy - pad), min(h, hy + pad)
    crop = gray_full[y0:y1, x0:x1]

    mask = cv2.inRange(crop, BODY_LO, BODY_HI)
    yy, xx = np.mgrid[:crop.shape[0], :crop.shape[1]]
    gx, gy = xx + x0, yy + y0
    for ox, oy in others:
        mask[np.hypot(gx - ox, gy - oy) < 1.1 * R] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, BODY_OPEN)

    dist = np.hypot(gx - hx, gy - hy)
    n, labels = cv2.connectedComponents(mask)
    best, best_count = None, 0
    for lab in range(1, n):
        m = labels == lab
        if dist[m].max() < BODY_MIN_EXT * R:
            continue
        touch = np.count_nonzero(m & (dist > 1.2 * R) & (dist < 2.0 * R))
        if touch < BODY_MIN_TOUCH:
            continue
        if m.sum() > best_count:
            best, best_count = m, int(m.sum())

    if best is None:
        return None
    body = best & (dist > 1.2 * R)
    ys_, xs_ = np.nonzero(body)
    return np.column_stack([xs_ + x0, ys_ + y0]).astype(np.float64)


class Target:
    def __init__(self, c, now, hit):
        self.c = c
        self.last_seen = now
        self.samples = deque([hit], maxlen=6)   # must be set before .hit is used
        self.body_checked = False
        self.born = now
        self.admit_lead = self.hit - now         # seconds left when admitted

    @property
    def hit(self):
        return float(np.median(self.samples))


class Slider:
    def __init__(self, head, pts, now):
        self.head = head
        self.pts = pts
        self.dist = np.hypot(pts[:, 0] - head[0], pts[:, 1] - head[1])
        self.t0 = now
        self.dur = float(self.dist.max()) / SLIDER_PX_PER_S + 0.1

    def target(self, now):
        d = SLIDER_PX_PER_S * (now - self.t0)
        i = int(np.argmin(np.abs(self.dist - d)))
        return self.pts[i]

    def done(self, now):
        return now - self.t0 > self.dur


def drop_reason(o, now):
    if now - o.last_seen >= LOST_AFTER:
        return "lost"
    if now >= o.hit + GRACE:
        return "hit"
    return None


def snapshot():
    """Save an annotated frame to check R against the real circles."""
    sct = mss.mss()
    raw = np.asarray(sct.grab(MONITOR))
    gray = cv2.cvtColor(raw, cv2.COLOR_BGRA2GRAY)
    gray_h = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    centers = detect_circles(gray_h)
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for cx, cy in centers:
        cv2.circle(out, (cx, cy), int(round(R)), (0, 255, 0), 2)
        cv2.circle(out, (cx, cy), 3, (0, 0, 255), -1)
    cv2.imwrite("snap.png", out)
    print(f"R = {R:.1f} px (full-res). Detected {len(centers)} circles.")
    print("Saved snap.png. Green circles should sit on the hit circles.")


def main():
    mouse = Controller()
    sct = mss.mss()

    print("Bot running. Press 'q' to stop.")
    objects = []
    slider = None
    current = None
    cursor = np.array(mouse.position, dtype=float)
    t_prev = time.perf_counter()
    last_dbg = 0.0
    admitted_window = 0
    mode = "idle"

    while not keyboard.is_pressed("q"):
        now = time.perf_counter()
        frame_ms = (now - t_prev) * 1000

        raw = np.asarray(sct.grab(MONITOR))
        gray = cv2.cvtColor(raw, cv2.COLOR_BGRA2GRAY)
        gray_h = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        centers = detect_circles(gray_h)

        # 1. Match detections to existing objects and refine hit times
        taken = set()
        rings_this_frame = 0
        for o in objects:
            best, best_d = None, 0.6 * R
            for i, c in enumerate(centers):
                if i in taken:
                    continue
                d = np.hypot(c[0] - o.c[0], c[1] - o.c[1])
                if d < best_d:
                    best, best_d = i, d
            if best is not None:
                taken.add(best)
                o.c = centers[best]
                o.last_seen = now

            others = [c for c in centers if c != o.c]
            r = find_ring(gray, o.c[0], o.c[1], others,
                          predicted_radius(o.hit, now))
            if r is not None:
                rings_this_frame += 1
                o.samples.append(hit_from_radius(r, now))

        # 2. Admit new circles whose approach ring is visible
        for i, c in enumerate(centers):
            if i in taken:
                continue
            others = [cc for cc in centers if cc != c]
            r = find_ring(gray, c[0], c[1], others)
            if r is not None and r >= ADMIT_MIN_R * R:
                objects.append(Target(c, now, hit_from_radius(r, now)))
                admitted_window += 1

        # 3. Drop objects that were hit or lost, and log why
        kept = []
        for o in objects:
            why = drop_reason(o, now)
            if why is None:
                kept.append(o)
            elif DEBUG:
                print(f"  drop {why}: admit_lead={o.admit_lead:+.3f}s "
                      f"age={now - o.born:.2f}s at={o.c}")
        objects = kept
        if current is not None and current not in objects:
            current = None

        # 4. Move toward the locked target
        if slider is not None and slider.done(now):
            slider = None

        if slider is not None:
            mode = "slider"
            tgt = slider.target(now) + np.array([MONITOR["left"], MONITOR["top"]])
            cursor += (tgt - cursor) * min(1.0, SLIDER_GAIN * (now - t_prev))
            mouse.position = (int(cursor[0]), int(cursor[1]))
        else:
            cands = [o for o in objects if o.hit >= now - 0.02]
            if cands:
                earliest = min(cands, key=lambda o: o.hit)
                if current is None or current not in cands or \
                        earliest.hit < current.hit - SWITCH_MARGIN:
                    current = earliest
                o = current
                mode = "circle"

                if ENABLE_SLIDERS and not o.body_checked and now >= o.hit - 0.02:
                    o.body_checked = True
                    others = [c for c in centers if c != o.c]
                    pts = slider_body(gray, o.c, others)
                    if pts is not None:
                        slider = Slider(o.c, pts, now)
                        objects.remove(o)
                        current = None
                        mode = "slider"

                if slider is None:
                    target = np.array([MONITOR["left"] + o.c[0],
                                       MONITOR["top"] + o.c[1]], dtype=float)
                    remaining = max(o.hit - MOUSE_LAG - now, 1e-3)
                    dt = now - t_prev
                    cursor += (target - cursor) * min(1.0, dt / remaining)
                    mouse.position = (int(cursor[0]), int(cursor[1]))
            else:
                current = None
                mode = "idle"

        if DEBUG and now - last_dbg > 1.0:
            print(f"detected={len(centers):2d}  tracked={len(objects):2d}  "
                  f"rings={rings_this_frame:2d}  admitted={admitted_window:2d}  "
                  f"mode={mode}  frame={frame_ms:4.0f}ms")
            last_dbg = now
            admitted_window = 0

        t_prev = now


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--snap":
        snapshot()
    else:
        main()