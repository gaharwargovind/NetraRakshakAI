import cv2

# Disable OpenCV multi-threading during pytest runs to prevent process hangs on macOS
cv2.setNumThreads(0)
