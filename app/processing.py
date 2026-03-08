import cv2 as cv
import numpy as np

from config import (
    BLUE_LOWER, BLUE_UPPER,
    RED_LOWER1, RED_UPPER1, RED_LOWER2, RED_UPPER2,
    MORPH_KERNEL, EDGE_KERNEL,
)

def get_colour_mask(frame, colour):
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    if colour == "red":
        mask = cv.bitwise_or(
            cv.inRange(hsv, RED_LOWER1, RED_UPPER1),
            cv.inRange(hsv, RED_LOWER2, RED_UPPER2),
        )
    elif colour == "blue":
        mask = cv.inRange(hsv, BLUE_LOWER, BLUE_UPPER)

    mask = cv.morphologyEx(mask, cv.MORPH_OPEN,  MORPH_KERNEL)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, MORPH_KERNEL)
    return mask

def sobel_edges(channel):
    gx  = cv.Sobel(channel, cv.CV_64F, 1, 0, ksize=3)
    gy  = cv.Sobel(channel, cv.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    mag = np.uint8(mag / mag.max() * 255)
    return cv.morphologyEx(mag, cv.MORPH_CLOSE, EDGE_KERNEL)

def colour_sobel(frame):
    filtered = cv.bilateralFilter(frame, 9, 75, 75)
    b, g, r  = cv.split(filtered)
    return cv.merge([sobel_edges(b), sobel_edges(g), sobel_edges(r)])