import numpy as np
import os
import matplotlib.pyplot as plt

from numpy.random import multivariate_normal as mvnrnd
from numpy.random import poisson
from scipy.spatial.transform import Rotation as Rt

from FuncTools import *
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

import os, sys
if not os.path.join(os.path.dirname(__file__), '../') in sys.path:
    sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import car_model

# Load measurement data
radar_point = np.load(os.path.join(os.path.dirname(__file__),\
            '../data/turn_around/radar/10.0-0.npy'), allow_pickle=True)
labels = np.load(os.path.join(os.path.dirname(__file__),\
            '../data/turn_around/labels.npy'), allow_pickle=True)

time_interval = 0.05

# nearly constant velocity model 
H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]])
Ar = np.array([[1, 0, 0, time_interval, 0, 0], \
                [0, 1, 0, 0, time_interval, 0], \
                [0, 0, 1, 0, 0, time_interval], \
                [0, 0, 0, 1, 0, 0], \
                [0, 0, 0, 0, 1, 0], \
                [0, 0, 0, 0, 0, 1]])
Ap = np.eye(6)

Ch = np.diag([1/4, 1/4, 1/4]) # covariance of the multiplicative noise
Cv = np.diag([0.5, 0.5, 0.5]) # covariance of the measurement noise
Cwr = np.diag([0.1, 0.1, 0.1, 0.5, 0.5, 0.5]) # covariance of the process noise for the kinematic state
Cwp = np.diag([0.001, 0.001, 0.001, 0.01, 0.01, 0.01]) #covariance of the process noise for the shape parameters

## Prior
pos0 = labels[0]['vehicle_pos'][0]
v0 = labels[0]['velocity'][0]
quat0 = labels[0]['vehicle_quats'][0]
euler0 = Rt.from_quat(quat0).as_euler('xyz', degrees=False)

r = np.array([pos0[0], pos0[1], pos0[2], v0[0], v0[1], v0[2]]).reshape([6,1])
p = np.array([euler0[0], euler0[1], euler0[2], 1, 1, 1]).reshape([6,1])

Cr = np.diag([2, 2, 2, 2, 2, 2])
Cp = np.diag([0.2, 0.2, 0.2, 1, 1, 1])

## Filtering
res=[]

showX,showY,showZ=[],[],[]

u = np.linspace(0, 2 * np.pi, 20)
v = np.linspace(0, np.pi, 20)

for i in range(len(radar_point)):
    y = np.array(radar_point[i]).T
    ## Update
    r, p, Cr, Cp = measurement_update_3d(y, H, r, p, Cr, Cp, Ch, Cv)
    now=dict()
    now['r']=r
    now['Cr']=Cr
    now['p']=p
    now['Cp']=Cp
    res.append(now)

    ## Generate ellipse properties
    R = Rt.from_euler('xyz',p[:3,0].reshape(3)).as_matrix()
    x_radius, y_radius, z_radius = p[3:,0]
    center=r[:3,0]

    x = x_radius * np.outer(np.cos(u), np.sin(v))
    y = y_radius * np.outer(np.sin(u), np.sin(v))
    z = z_radius * np.outer(np.ones_like(u), np.cos(v))

    points = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)

    points_rotated = (R @ points.T).T
    x_rotated, y_rotated, z_rotated = np.split(points_rotated, 3, axis=1)
    x_translated = x_rotated + center[0]
    y_translated = y_rotated + center[1]
    z_translated = z_rotated + center[2]

    showX.append(x_translated)
    showY.append(y_translated)
    showZ.append(z_translated)

    ## Predict next state
    r, p, Cr, Cp = time_update(r, p, Cr, Cp, Ar, Ap, Cwr, Cwp)

np.save(os.path.join(os.path.dirname(__file__),\
                'MEM-EKFstar_result.npy'),res,allow_pickle=True)

## Show results
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Ground truth extend
v = np.zeros(car_model.keypoints.shape)
skeleton = [v[item] for item in car_model.skeleton_24]

# Creating Cube Objects
framed = Line3DCollection(skeleton, colors='k', linewidths=0.2, linestyles='-')
ax.add_collection3d(framed)

# Set original ellipse object
x_radius = 0
y_radius = 0
z_radius = 0
center = [0, 0, 0]
x = x_radius * np.outer(np.cos(u), np.sin(v))
y = y_radius * np.outer(np.sin(u), np.sin(v))
z = z_radius * np.outer(np.ones_like(u), np.cos(v))
ellipse = ax.plot_surface(x, y, z, color='blue', alpha=0.25)

# Set coordinate axis range
ax.set_xlim(-10, 10)
ax.set_ylim(-30, 10)
ax.set_zlim(0, 60)
ax.set_box_aspect((1, 2, 3))

def show(frame):
    global y, H, r, p, Cr, Cp, Ch, Cv, ellipse, framed

    ## Draw ground truth
    v = np.array(labels[frame]['keypoints_world_all'][0])[:, 1:4]
    pos_gt = np.array(labels[frame]['vehicle_pos'][0])
    skeleton = [v[item] for item in car_model.skeleton_24]
    framed.set_segments(skeleton)

    ## Draw the estimated ellipse
    ellipse.remove()
    ellipse = ax.plot_surface(showX[frame].reshape([20,20]),
                             showY[frame].reshape([20,20]), 
                             showZ[frame].reshape([20,20]), color='blue', alpha=0.25)
    
    ## Car in the middle of the window
    ax.set_xlim(pos_gt[0] - 4, pos_gt[0] + 4)
    ax.set_ylim(pos_gt[1] - 4, pos_gt[1] + 4)
    ax.set_zlim(pos_gt[2] - 4, pos_gt[2] + 4)
    ax.set_box_aspect((1, 1, 1))
    
    return ax,

ani = animation.FuncAnimation(fig, show, frames=range(len(labels)), interval=50)

plt.show()

ax.azim, ax.elev = -79.66314935064936, 5.649350649350639
# ani.save('animation.gif', writer='pillow', dpi=100)