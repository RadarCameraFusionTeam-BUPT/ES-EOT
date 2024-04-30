import numpy as np
import os, sys
import matplotlib.pyplot as plt
from FuncTools import quaternion_to_rotation_matrix, K, quaternion_to_rotation_matrix
from config import *

from scipy.spatial.transform import Rotation as Rt

from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay, ConvexHull
import torch as th

if not os.path.join(os.path.dirname(__file__), '../') in sys.path:
    sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import car_model

# Get ground truth and tracking result
labels = np.load(os.path.join(os.path.dirname(__file__),\
            '../data/turn_around/labels.npy'), allow_pickle=True)
res = np.load(os.path.join(os.path.dirname(__file__),\
                './GPEOT_result.npy'),allow_pickle=True)

## Show results
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Ground truth extend
v = np.zeros(car_model.keypoints.shape)
skeleton = [v[item] for item in car_model.skeleton_24]

# Creating Cube Objects
framed = Line3DCollection(skeleton, colors='k', linewidths=0.2, linestyles='-')
ax.add_collection3d(framed)

# Initialized Gaussian Process model
num_basis_points = len(res[0]['u_f'])
shape3D=Poly3DCollection([np.zeros([4, 3]) for i in range(num_basis_points)], alpha=0.5, linewidths=0.1,  edgecolors='k', linestyles=':')
ax.add_collection3d(shape3D)

u_f = th.tensor(res[0]['u_f'], dtype=th.float64)

theta_f = u_f[:, 0]
phi_f = u_f[:, 1]
x_f = th.cos(phi_f) * th.cos(theta_f)
y_f = th.cos(phi_f) * th.sin(theta_f)
z_f = th.sin(phi_f)
p_f = th.stack((x_f, y_f, z_f), dim = 1)

K_uf_uf = K(u_f, u_f)
H_f = K_uf_uf @ th.inverse(K_uf_uf)

# Set coordinate axis range
ax.set_xlim(-10, 10)
ax.set_ylim(-30, 10)
ax.set_zlim(0, 60)
ax.set_box_aspect((1, 2, 3))

def show(frame):
    global framed, shape3D

    ## Draw ground truth
    v = np.array(labels[frame]['keypoints_world_all'][0])[:, 1:4]
    pos_gt = np.array(labels[frame]['vehicle_pos'][0])
    skeleton = [v[item] for item in car_model.skeleton_24]
    framed.set_segments(skeleton)

    ## Draw the estimated shape
    pos = res[frame]['pos']
    quat = th.tensor(res[frame]['quat'], dtype=th.float64)
    f = th.tensor(res[frame]['f'], dtype=th.float64)

    R=quaternion_to_rotation_matrix(quat).numpy()

    len_f = H_f @ f
    extend = p_f * len_f.reshape((-1, 1))
    extend = (R @ extend.numpy().T).T + pos

    faces=ConvexHull(extend).simplices

    shape3D.set_verts(extend[faces])

    ## Car in the middle of the window
    ax.set_xlim(pos_gt[0] - 4, pos_gt[0] + 4)
    ax.set_ylim(pos_gt[1] - 4, pos_gt[1] + 4)
    ax.set_zlim(pos_gt[2] - 10, pos_gt[2] + 10)
    ax.set_box_aspect((8, 8, 20))
    
    return ax,

ani = animation.FuncAnimation(fig, show, frames=range(len(res)), interval=50)

plt.show()

# ax.azim, ax.elev = -79.66314935064936, 5.649350649350639
# ani.save('animation.gif', writer='pillow', dpi=100)