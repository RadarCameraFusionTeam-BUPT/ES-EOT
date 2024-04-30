import numpy as np
import os
import matplotlib.pyplot as plt
from config import *

from scipy.spatial.transform import Rotation as Rt

from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay, ConvexHull

if not os.path.join(os.path.dirname(__file__), '../') in sys.path:
    sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import car_model

# Get ground truth and tracking result
# labels = np.load(os.path.join(os.path.dirname(__file__),\
#             '../data/turn_around/labels.npy'), allow_pickle=True)
# res = np.load(os.path.join(os.path.dirname(__file__),\
#                 './ES-EOT_result.npy'), allow_pickle=True)

labels = np.load(os.path.join(os.path.dirname(__file__),\
            '../data/turn_around/labels.npy'), allow_pickle=True)
res = np.load(os.path.join(os.path.dirname(__file__),\
                './ES-EOT_result.npy'), allow_pickle=True)

## Show results
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Ground truth extend
v = np.zeros(car_model.keypoints.shape)
skeleton = [v[item] for item in car_model.skeleton_24]

# Creating Cube Objects
framed = Line3DCollection(skeleton, colors='k', linewidths=0.2, linestyles='-')
ax.add_collection3d(framed)

# Initialized point position
scatter_keypoints = ax.scatter(np.zeros(N_T), np.zeros(N_T), np.zeros(N_T), c='b')
scatrer_base = ax.scatter(np.zeros(N_T), np.zeros(N_T), np.zeros(N_T), c='r', s=5)

# Set coordinate axis range
ax.set_xlim(-10, 10)
ax.set_ylim(-30, 10)
ax.set_zlim(0, 80)
ax.set_box_aspect((1, 2, 3))

def show(frame):
    global framed

    ## Draw ground truth
    v = np.array(labels[frame]['keypoints_world_all'][0])[:, 1:4]
    pos_gt = np.array(labels[frame]['vehicle_pos'][0])
    skeleton = [v[item] for item in car_model.skeleton_24]
    framed.set_segments(skeleton)

    ## Draw the estimated shape
    x_ref = res[frame]['x_ref']
    pos = x_ref[:3]
    theta = x_ref[4:7]
    mu = res[frame]['mu']
    u = mu[:, 3:6]
    base = mu[:, :3]
    m = res[frame]['m']

    R = Rt.from_rotvec(theta).as_matrix()
    u = (R @ u.T).T + pos
    base = (R @ base.T).T + pos

    alp = m / m.sum()

    scatter_keypoints._offsets3d = (u[:,0], u[:,1], u[:,2])
    scatter_keypoints.set_sizes(alp * 300)
    scatrer_base._offsets3d = (base[:,0], base[:,1], base[:,2])
    # scatrer_base.set_sizes(alp * 300)

    ## Car in the middle of the window
    ax.set_xlim(pos[0] - 4, pos[0] + 4)
    ax.set_ylim(pos[1] - 4, pos[1] + 4)
    ax.set_zlim(pos[2] - 10, pos[2] + 10)
    ax.set_box_aspect((8, 8, 20))
    
    return ax,

ani = animation.FuncAnimation(fig, show, frames=range(len(res)), interval=50)

plt.show()

# ax.azim, ax.elev = -79.66314935064936, 5.649350649350639
# ani.save('animation.gif', writer='pillow', dpi=100)