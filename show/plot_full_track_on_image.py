import os
import numpy as np
import sys
import cv2
from utils import *

# adjustable parameters
scenario = 'turn_around'
frame_show = 30
image_show_size = (3840, 2160)

# set path
video_path = os.path.join(os.path.dirname(__file__),\
            '../data/{}/vision/output.mp4'.format(scenario))

labels = np.load(os.path.join(os.path.dirname(__file__),\
            '../data/{}/labels.npy'.format(scenario)), allow_pickle=True)

keypoints_det = np.load(os.path.join(os.path.dirname(__file__),\
            '../data/{}/vision/output-keypoints.npy'.format(scenario)), allow_pickle=True)

# data range of the vehicle during normal driving conditions
data_frame_begin = 8
data_frame_end = len(labels) - 5

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error opening video stream or file")
    exit(0)

now_frame = 0
frame = None
while now_frame < frame_show and cap.isOpened():
    _, frame = cap.read()
    now_frame += 1

if frame is None:
    print("Error reading frame")
    exit(0)

car_positions = np.array([frame['vehicle_pos'][0] for frame in labels])
pixel_coordinates = (K @ car_positions.T).T
pixel_coordinates = pixel_coordinates / pixel_coordinates[:, 2].reshape([-1, 1])
pixel_coordinates = pixel_coordinates[:, :2].astype(np.int32)

for i in range(data_frame_begin, data_frame_end):
    cv2.line(frame, pixel_coordinates[i], pixel_coordinates[i + 1], (255, 0, 0), 20)

for pix in keypoints_det[frame_show]['keypoints'][0]:
    if int(pix[0]) in corner_id:
        cv2.circle(frame, (int(pix[1]), int(pix[2])), 10, (255, 0, 255), -1)
    elif int(pix[0]) in skeleton_knots_id:
        cv2.circle(frame, (int(pix[1]), int(pix[2])), 10, (0, 255, 0), -1)

frame = cv2.resize(frame, image_show_size)
# cv2.imshow("frame", frame)
# cv2.waitKey(0)
cv2.imwrite("full_track_on_image.jpg", frame)

cap.release()
cv2.destroyAllWindows()