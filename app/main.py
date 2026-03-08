import cv2 as cv
import numpy as np
import time

from config import CAMERA_WIDTH, CAMERA_HEIGHT, WINDOW_NAME
from processing import colour_sobel
from detection import detect_cubes, detect_bean_height, detect_bean_ellipse
from calibration import interpolate_ppcm
from drawing import draw_cubes, draw_bean_h, draw_bean_e, draw_hud

cap_front = cv.VideoCapture(0)   # front view
cap_top   = cv.VideoCapture(1)   # top-down

if not cap_front.isOpened():
    print("Could not open camera 0 (front)")
    exit(1)
if not cap_top.isOpened():
    print("Could not open camera 1 (top-down)")
    exit(1)

for cap in (cap_front, cap_top):
    cap.set(cv.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

view_mode  = 0                          # 0 = original, 1 = edges
view_names = ["Original", "Edge Detection"]
ppcm_front = None
ppcm_top   = None

avg_window_start    = time.time()
height_samples      = []
ellipse_maj_samples = []
ellipse_min_samples = []

avg_height_cm = None
avg_major_cm  = None
avg_minor_cm  = None

# Main loop

while True:
    ret_f, frame_front = cap_front.read()
    ret_t, frame_top   = cap_top.read()
    if not ret_f or not ret_t:
        break

    # Match resolutions so we can stack them side-by-side
    frame_top = cv.resize(frame_top, (frame_front.shape[1], frame_front.shape[0]))

    #Edge images
    edges_front = colour_sobel(frame_front)
    edges_top   = colour_sobel(frame_top)

    # Detect reference cubes
    cubes_front, ppcm_front = detect_cubes(frame_front, "blue",  ppcm_front)
    cubes_top,   ppcm_top   = detect_cubes(frame_top,   "red",   ppcm_top)

    # Detect bean
    bean_front = detect_bean_height(frame_front, cubes_front)
    bean_top   = detect_bean_ellipse(frame_top,  cubes_top)

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

    if bean_h_cm:
        height_samples.append(bean_h_cm)
    if bean_axes_cm:
        ellipse_maj_samples.append(bean_axes_cm[0])
        ellipse_min_samples.append(bean_axes_cm[1])

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

    # Annotate edge view
    disp_edge_f = edges_front.copy()
    disp_edge_t = edges_top.copy()
    draw_cubes(disp_edge_f, cubes_front)
    draw_cubes(disp_edge_t, cubes_top)
    draw_bean_h(disp_edge_f, bean_front, bean_h_cm)
    draw_bean_e(disp_edge_t, bean_top,   bean_axes_cm)

    # Annotate original view
    draw_bean_h(frame_front, bean_front, bean_h_cm)
    draw_bean_e(frame_top,   bean_top,   bean_axes_cm)

    # Final display
    if view_mode == 1:
        display = np.hstack((disp_edge_f, disp_edge_t))
    else:
        display = np.hstack((frame_front, frame_top))

    draw_hud(
        display, view_names[view_mode], ppcm_front, ppcm_top,
        bean_h_cm, interp_f, bean_axes_cm, bean_top,
        avg_height_cm, avg_major_cm, avg_minor_cm,
        cubes_front, cubes_top, bean_front,
    )
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
