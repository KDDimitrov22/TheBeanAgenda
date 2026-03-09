#Project constants and tuning values.

import numpy as np

# Camera

CAMERA_WIDTH  = 640
CAMERA_HEIGHT = 480

# Reference cubes

# Cube face size (cm)
REFERENCE_CM = 1.0

# Cube contour area limits (px²)
CUBE_AREA_MIN = 200
CUBE_AREA_MAX = 5000

# Max aspect ratio for cube-like contours
CUBE_MAX_ASPECT = 1.5

# HSV color ranges for cube detection

BLUE_LOWER = np.array([90, 50, 50])
BLUE_UPPER = np.array([130, 255, 255])

# Red wraps around H=0/180, so it uses two ranges
RED_LOWER1 = np.array([0, 50, 50])
RED_UPPER1 = np.array([10, 255, 255])
RED_LOWER2 = np.array([170, 50, 50])
RED_UPPER2 = np.array([180, 255, 255])

# Morphology kernels

MORPH_KERNEL = np.ones((5, 5), np.uint8)   # For color-mask cleanup
EDGE_KERNEL  = np.ones((2, 2), np.uint8)   # For Sobel edge cleanup
BEAN_KERNEL  = np.ones((3, 3), np.uint8)   # For bean contour cleanup

# Bean detection

# Bean contour area limits (px²)
BEAN_AREA_MIN = 100
BEAN_AREA_MAX = 15000

# Acceptable bean aspect-ratio range
BEAN_ASPECT_MIN = 1.3
BEAN_ASPECT_MAX = 5.0

# Horizontal margin from cube centers when building the search band (px)
BEAN_SEARCH_MARGIN = 30

# Display

WINDOW_NAME = "Dual Camera View"
