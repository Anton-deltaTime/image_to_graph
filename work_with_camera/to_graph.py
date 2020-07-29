import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import numpy
import cv2 as cv

cap = cv.VideoCapture(0, apiPreference=cv.CAP_DSHOW)
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


def make_data(frame_):
    rows, cels = frame_.shape[:2]
    ax = numpy.arange(0, cels, 1)
    ay = numpy.arange(0, rows, -1)
    # x = numpy.arange(-10, 10, 0.1)
    xgrid, ygrid = numpy.meshgrid(ax, ay)
    
    gray = cv.cvtColor(frame_, cv.COLOR_BGR2GRAY)

    # az = gray[:, :]
    az = cv.bitwise_not(gray)
    zgrid = az[:, :] // 2
    return xgrid, ygrid, zgrid


x, y, z = make_data(frame)

fig = plt.figure()
axes = Axes3D(fig)
axes.set_zlim3d([0, 120])

axes.plot_surface(x, y, z)


def update(i):
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    az = cv.bitwise_not(gray)
    zgrid = az[:, :] // 2
    # surface.set_3d_properties(zgrid)
    return zgrid


# ani = FuncAnimation(fig, update, repeat=False)

plt.show()
