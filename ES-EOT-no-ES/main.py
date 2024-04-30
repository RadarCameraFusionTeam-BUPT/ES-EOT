from config import *
from FuncTools import *
import numpy as np
import time
import os

# Load measurement data
radar_point = np.load(os.path.join(os.path.dirname(__file__),\
            '../data/turn_around/radar/10.0-0.npy'), allow_pickle=True)
labels = np.load(os.path.join(os.path.dirname(__file__),\
            '../data/turn_around/labels.npy'), allow_pickle=True)
keypoints_det = np.load(os.path.join(os.path.dirname(__file__),\
            '../data/turn_around/vision/output-keypoints.npy'), allow_pickle=True)

# Prior
pos = labels[0]['vehicle_pos'][0]
v = np.linalg.norm(labels[0]['velocity'][0])
quat = labels[0]['vehicle_quats'][0]
theta = Rt.from_quat(quat).as_rotvec()

x_ref = np.array([pos[0], pos[1], pos[2], v, theta[0], theta[1], theta[2], 0, 0, 0, 1, 1])

dx = np.zeros(len(x_ref))
P = np.eye(len(x_ref))

m = np.ones(N_T)

mu = np.random.normal(0, 1, (N_T, 9))
# mu[:, :3] = control_points
Sigma = np.tile(np.identity(mu.shape[1]), (N_T, 1, 1))

# Concatenate all state
Theta = State(x_ref, dx, P, m, mu, Sigma)

res = []

for i in range(len(radar_point)):
    z_r = np.array(radar_point[i])
    z_c = np.array(keypoints_det[i]['keypoints'][0], dtype=np.float64)

    print(i)

    Theta = update(Theta, z_r, z_c)

    now = dict()
    now['x_ref'] = Theta.x_ref.copy()
    now['P'] = Theta.P.copy()
    now['m'] = Theta.m.copy()
    now['mu'] = Theta.mu.copy()
    now['Sigma'] = Theta.Sigma.copy()
    res.append(now)

    Theta = predict(Theta)
    

np.save(os.path.join(os.path.dirname(__file__),\
        'ES-EOT_result.npy'), res, allow_pickle = True)