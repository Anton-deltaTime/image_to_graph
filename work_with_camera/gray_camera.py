import numpy as np
import cv2 as cv

USE_CAMERA = 0

cap = cv.VideoCapture(USE_CAMERA, apiPreference=cv.CAP_DSHOW)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # gray = cv.bitwise_not(gray)
    # rows, cols = gray.shape
    # frame_ = np.zeros((rows, cols), np.uint8)
    # frame_[:, :] = gray[:, :] // 2

    cv.imshow('frame', gray)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
