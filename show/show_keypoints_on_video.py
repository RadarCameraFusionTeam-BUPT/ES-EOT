import os
import numpy as np
import sys
import cv2

# adjustable parameters
scenario = 'turn_around'

# set path
keypoints_det = np.load(os.path.join(os.path.dirname(__file__),\
            '../data/{}/vision/output-keypoints.npy'.format(scenario)), allow_pickle=True)

video_path = os.path.join(os.path.dirname(__file__),\
            '../data/{}/vision/output.mp4'.format(scenario))

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error opening video stream or file")

video_out = cv2.VideoWriter('video_out.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (960, 540))

idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    for i in range(len(keypoints_det[idx]['keypoints'][0])):
        cv2.circle(frame, (int(keypoints_det[idx]['keypoints'][0][i][1]), int(keypoints_det[idx]['keypoints'][0][i][2])),\
                    10, (0, 255, 0), -1)

    frame = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)
    video_out.write(frame)
    # cv2.imshow('frame', frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

    idx += 1

video_out.release()
cap.release()
cv2.destroyAllWindows()