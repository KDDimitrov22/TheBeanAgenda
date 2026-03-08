import cv2 as cv
import numpy as np
import time

# HSV colour ranges for cube detection
BLUE_LOWER  = np.array([90, 50, 50])
BLUE_UPPER  = np.array([130, 255, 255])
RED_LOWER1  = np.array([0, 50, 50])
RED_UPPER1  = np.array([10, 255, 255])
RED_LOWER2  = np.array([170, 50, 50])
RED_UPPER2  = np.array([180, 255, 255])

MORPH_KERNEL = np.ones((5, 5), np.uint8)
EDGE_KERNEL  = np.ones((2, 2), np.uint8)

WINDOW_NAME = "Dual Camera View"

# Reference cube size
REFERENCE_CM = 1.0

cap_front = cv.VideoCapture(0)   # front view
cap_top   = cv.VideoCapture(1)   # top-down

if not cap_front.isOpened():
    print("Error: Could not open camera 0 (front)")
    exit(1)
if not cap_top.isOpened():
    print("Error: Could not open camera 1 (top-down)")
    exit(1)

for cap in (cap_front, cap_top):
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

view_mode  = 0                          # 0 = original, 1 = edges
view_names = ["Original", "Edge Detection"]
ppcm_front = None
ppcm_top   = None

avg_window_start = time.time()
height_samples   = []
ellipse_maj_samples = []
ellipse_min_samples = []

avg_height_cm   = None
avg_major_cm    = None
avg_minor_cm    = None

# Colour masking

def get_colour_mask(frame, colour):
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    if colour == "red":
        mask = cv.bitwise_or(
            cv.inRange(hsv, RED_LOWER1, RED_UPPER1),
            cv.inRange(hsv, RED_LOWER2, RED_UPPER2),
        )
    else:
        mask = cv.inRange(hsv, BLUE_LOWER, BLUE_UPPER)

    # Open removes small noise, close fills small holes
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN,  MORPH_KERNEL)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, MORPH_KERNEL)
    return mask

# Edge detection

def sobel_edges(channel):
    gx = cv.Sobel(channel, cv.CV_64F, 1, 0, ksize=3)
    gy = cv.Sobel(channel, cv.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    mag = np.uint8(mag / mag.max() * 255)
    return cv.morphologyEx(mag, cv.MORPH_CLOSE, EDGE_KERNEL)


def colour_sobel(frame):
    filtered = cv.bilateralFilter(frame, 9, 75, 75)
    b, g, r = cv.split(filtered)
    return cv.merge([sobel_edges(b), sobel_edges(g), sobel_edges(r)])

# Cube detection & calibration

def detect_cubes(frame, colour, cached_ppcm=None):
    mask = get_colour_mask(frame, colour)
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    measurements = []
    sizes = []  # pixel sizes of candidate cubes (for calibration)

    for cnt in contours:
        area = cv.contourArea(cnt)
        if area < 200 or area > 5000:
            continue

        rect = cv.minAreaRect(cnt)
        bw, bh = rect[1]
        if bw == 0 or bh == 0:
            continue

        if max(bw, bh) / min(bw, bh) >= 1.5:
            continue

        sizes.append(min(bw, bh))
        cx, cy = int(rect[0][0]), int(rect[0][1])
        measurements.append((bh, bw, cx, cy, int(bw), int(bh), rect[2]))

    ppcm = cached_ppcm
    if ppcm is None and sizes:
        ppcm = np.median(sizes) / REFERENCE_CM
    if ppcm is None or ppcm < 5:
        ppcm = 50

    measurements.sort(key=lambda m: m[2])
    return measurements[:2], ppcm

# Depth-interpolated calibration

def interpolate_ppcm(cubes, target_y):
    if len(cubes) < 2:
        return None

    # Each cube's apparent size
    size_a = min(cubes[0][0], cubes[0][1])
    size_b = min(cubes[1][0], cubes[1][1])
    y_a, y_b = cubes[0][3], cubes[1][3]
    ppcm_a = size_a / REFERENCE_CM
    ppcm_b = size_b / REFERENCE_CM

    if abs(y_a - y_b) < 1:
        return (ppcm_a + ppcm_b) / 2

    # Linear interpolation
    t = (target_y - y_a) / (y_b - y_a)
    result = ppcm_a + t * (ppcm_b - ppcm_a)
    return max(result, min(ppcm_a, ppcm_b))

# Bean detection

def _find_bean_contour(frame, cubes):
    if len(cubes) < 2:
        return None

    h_frame, w_frame = frame.shape[:2]

    x_left  = min(cubes[0][2], cubes[1][2])
    x_right = max(cubes[0][2], cubes[1][2])
    margin  = 30
    roi_x1  = max(0, x_left + margin)
    roi_x2  = min(w_frame, x_right - margin)

    if roi_x2 <= roi_x1:
        roi_x1 = max(0, x_left - margin)
        roi_x2 = min(w_frame, x_right + margin)

    # Build a combined binary image
    grey    = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(grey, (7, 7), 0)
    thresh  = cv.adaptiveThreshold(
        blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY_INV, 31, 8,
    )
    edges    = cv.Canny(blurred, 30, 100)
    combined = cv.bitwise_or(thresh, edges)

    # Morphological cleanup
    k = np.ones((3, 3), np.uint8)
    combined = cv.morphologyEx(combined, cv.MORPH_CLOSE, k, iterations=2)
    combined = cv.morphologyEx(combined, cv.MORPH_OPEN,  k, iterations=1)

    contours, _ = cv.findContours(combined, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    best, best_score = None, 0
    mid_x = (roi_x1 + roi_x2) / 2

    for cnt in contours:
        area = cv.contourArea(cnt)
        if area < 400 or area > 15000 or len(cnt) < 5:
            continue

        M = cv.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])

        if not (roi_x1 <= cx <= roi_x2):
            continue

        rw, rh = cv.minAreaRect(cnt)[1]
        if rw == 0 or rh == 0:
            continue
        if not (1.3 <= max(rw, rh) / min(rw, rh) <= 5.0):
            continue

        score = area * (1.0 - abs(cx - mid_x) / (w_frame + 1))
        if score > best_score:
            best_score = score
            best = cnt

    return best


def detect_bean_height(frame, cubes):
    cnt = _find_bean_contour(frame, cubes)
    if cnt is None:
        return None
    x, y, w, h = cv.boundingRect(cnt)
    return (h, y, y + h, x + w // 2, y + h // 2)


def detect_bean_ellipse(frame, cubes):
    cnt = _find_bean_contour(frame, cubes)
    if cnt is None or len(cnt) < 5:
        return None
    (cx, cy), (a1, a2), angle = cv.fitEllipse(cnt)
    return (a1, a2, cx, cy, angle)

# Drawing helpers

def draw_cubes(img, cubes):
    for h_px, w_px, cx, cy, bw, bh, angle in cubes:
        box = cv.boxPoints(((cx, cy), (bw, bh), angle))
        box = np.asarray(box, dtype=np.int32)
        cv.polylines(img, [box], True, (0, 255, 0), 2)
        cv.putText(img, f"H:{h_px:.0f}px W:{w_px:.0f}px ({cx},{cy})",
                   (cx - 50, cy - 5), cv.FONT_HERSHEY_SIMPLEX, 0.35,
                   (0, 255, 0), 1)


def draw_bean_h(img, bean, height_cm=None):
    if bean is None:
        return
    h_px, top, bot, cx, cy = bean
    arm = 30  # half-length of the horizontal bracket lines

    cv.line(img, (cx - arm, top), (cx + arm, top), (0, 255, 255), 2)
    cv.line(img, (cx - arm, bot), (cx + arm, bot), (0, 255, 255), 2)
    cv.line(img, (cx, top), (cx, bot), (0, 255, 255), 1)
    cv.circle(img, (cx, cy), 3, (0, 255, 255), -1)

    label = f"Bean H: {h_px}px"
    if height_cm is not None:
        label += f" = {height_cm:.2f}cm"
    cv.putText(img, label, (cx - 60, top - 10),
               cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)


def draw_bean_e(img, ellipse, axes_cm=None):
    if ellipse is None:
        return
    a1, a2, cx, cy, angle = ellipse
    centre = (int(cx), int(cy))
    axes   = (int(a1 / 2), int(a2 / 2))

    cv.ellipse(img, centre, axes, angle, 0, 360, (0, 255, 255), 2)
    cv.circle(img, centre, 4, (0, 0, 255), -1)

    major, minor = max(a1, a2), min(a1, a2)
    if axes_cm:
        label = f"{major:.0f}x{minor:.0f}px = {axes_cm[0]:.2f}x{axes_cm[1]:.2f}cm"
    else:
        label = f"{major:.0f}x{minor:.0f}px"
    cv.putText(img, label, (centre[0] - 80, centre[1] - 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    cv.putText(img, f"@ ({cx:.0f},{cy:.0f})",
               (centre[0] - 40, centre[1] + int(major / 2) + 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

# HUD overlay (top-left info text)

def draw_hud(img, mode_name, ppcm_f, ppcm_t,
             bean_h_cm, interp_f, bean_axes_cm, bean_top,
             avg_h=None, avg_maj=None, avg_min=None,
             cubes_f=None, cubes_t=None, bean_front=None):

    cubes_f = cubes_f or []
    cubes_t = cubes_t or []
    y = 20
    cv.putText(img, f"{mode_name} (Sobel) | SPACE=view Q=quit",
               (10, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Calibration info
    if ppcm_f and ppcm_t:
        y += 20
        cv.putText(img,
            f"Blue: {ppcm_f:.1f}px/cm ({10/ppcm_f:.3f}mm/px) | "
            f"Red: {ppcm_t:.1f}px/cm ({10/ppcm_t:.3f}mm/px)",
            (10, y), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Cube detections
    if cubes_f:
        y += 20
        parts = [f"{h:.0f}x{w:.0f}px @({x},{cy})" for h, w, x, cy, *_ in cubes_f]
        cv.putText(img, f"Front cubes: {' | '.join(parts)}",
                   (10, y), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    if cubes_t:
        y += 20
        parts = [f"{h:.0f}x{w:.0f}px @({x},{cy})" for h, w, x, cy, *_ in cubes_t]
        cv.putText(img, f"Top cubes:   {' | '.join(parts)}",
                   (10, y), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # Bean height (front cam)
    if bean_front:
        y += 20
        txt = f"Bean H: {bean_front[0]}px"
        if bean_h_cm:
            txt += f" = {bean_h_cm:.2f}cm"
            if interp_f:
                txt += f" (interp {interp_f:.1f}px/cm)"
        if avg_h:
            txt += f"  avg: {avg_h:.2f}cm"
        cv.putText(img, txt, (10, y),
                   cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    # Bean ellipse (top cam)
    if bean_top:
        y += 20
        major = max(bean_top[0], bean_top[1])
        minor = min(bean_top[0], bean_top[1])
        txt = f"Bean top: {major:.0f}x{minor:.0f}px"
        if bean_axes_cm:
            txt += f" = {bean_axes_cm[0]:.2f}x{bean_axes_cm[1]:.2f}cm"
        if avg_maj and avg_min:
            txt += f"  avg: {avg_maj:.2f}x{avg_min:.2f}cm"
        txt += f" @({bean_top[2]:.0f},{bean_top[3]:.0f})"
        cv.putText(img, txt, (10, y),
                   cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

# Main loop

while True:
    ret_f, frame_front = cap_front.read()
    ret_t, frame_top   = cap_top.read()
    if not ret_f or not ret_t:
        break

    # Match resolutions
    frame_top = cv.resize(frame_top, (frame_front.shape[1], frame_front.shape[0]))

    edges_front = colour_sobel(frame_front)
    edges_top   = colour_sobel(frame_top)

    cubes_front, ppcm_front = detect_cubes(frame_front, "blue",  ppcm_front)
    cubes_top,   ppcm_top   = detect_cubes(frame_top,   "red",   ppcm_top)

    # Detect bean
    bean_front = detect_bean_height(frame_front, cubes_front)   # height
    bean_top   = detect_bean_ellipse(frame_top,  cubes_top)     # ellipse

    # Bean height in cm (front cam, depth-interpolated)
    bean_h_cm = None
    interp_f  = None
    if bean_front:
        if len(cubes_front) >= 2:
            interp_f = interpolate_ppcm(cubes_front, bean_front[4])
            if interp_f:
                bean_h_cm = bean_front[0] / interp_f
        elif ppcm_front:
            bean_h_cm = bean_front[0] / ppcm_front

    # Bean ellipse axes in cm (top cam, depth-interpolated)
    bean_axes_cm = None
    if bean_top:
        major_px = max(bean_top[0], bean_top[1])
        minor_px = min(bean_top[0], bean_top[1])
        if len(cubes_top) >= 2:
            interp_t = interpolate_ppcm(cubes_top, bean_top[3])
            if interp_t:
                bean_axes_cm = (major_px / interp_t, minor_px / interp_t)
        elif ppcm_top:
            bean_axes_cm = (major_px / ppcm_top, minor_px / ppcm_top)

    # Accumulate samples
    if bean_h_cm:
        height_samples.append(bean_h_cm)
    if bean_axes_cm:
        ellipse_maj_samples.append(bean_axes_cm[0])
        ellipse_min_samples.append(bean_axes_cm[1])

    # Compute averages every second
    now = time.time()
    if now - avg_window_start >= 1.0:
        if height_samples:
            avg_height_cm = np.mean(height_samples)
        if ellipse_maj_samples and ellipse_min_samples:
            avg_major_cm = np.mean(ellipse_maj_samples)
            avg_minor_cm = np.mean(ellipse_min_samples)
        height_samples.clear()
        ellipse_maj_samples.clear()
        ellipse_min_samples.clear()
        avg_window_start = now

    disp_edge_f = edges_front.copy()
    disp_edge_t = edges_top.copy()
    draw_cubes(disp_edge_f, cubes_front)
    draw_cubes(disp_edge_t, cubes_top)
    draw_bean_h(disp_edge_f, bean_front, bean_h_cm)
    draw_bean_e(disp_edge_t, bean_top,   bean_axes_cm)

    draw_bean_h(frame_front, bean_front, bean_h_cm)
    draw_bean_e(frame_top,   bean_top,   bean_axes_cm)

    # Compose final display
    if view_mode == 1:
        display = np.hstack((disp_edge_f, disp_edge_t))
    else:
        display = np.hstack((frame_front, frame_top))

    draw_hud(display, view_names[view_mode], ppcm_front, ppcm_top,
             bean_h_cm, interp_f, bean_axes_cm, bean_top,
             avg_height_cm, avg_major_cm, avg_minor_cm,
             cubes_front, cubes_top, bean_front)
    cv.imshow(WINDOW_NAME, display)

    # Keyboard input
    key = cv.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord(" "):
        view_mode = (view_mode + 1) % 2

cap_front.release()
cap_top.release()
cv.destroyAllWindows()