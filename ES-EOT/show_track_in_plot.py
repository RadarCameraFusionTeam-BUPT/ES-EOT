import numpy as np
import os
import matplotlib.pyplot as plt
from config import *
import cv2

from scipy.spatial.transform import Rotation as Rt

from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay, ConvexHull

if not os.path.join(os.path.dirname(__file__), '../') in sys.path:
    sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import car_model

# adjustable parameters
scenario = 'turn_around'
density = 10.0
idx = 0
show_frames = [30, 90, 150, 180, 240]

# Get ground truth and tracking result
labels = np.load(os.path.join(os.path.dirname(__file__),\
            '../data/{}/labels.npy'.format(scenario)), allow_pickle=True)
res = np.load(os.path.join(os.path.dirname(__file__),\
            './res/{}-{}-{}.npy'.format(scenario, density, idx)), allow_pickle=True)
radar_point = np.load(os.path.join(os.path.dirname(__file__),\
            '../data/{}/radar/{}-{}.npy'.format(scenario, density, idx)), allow_pickle=True)

# data range of the vehicle during normal driving conditions
data_frame_begin = 8
data_frame_end = len(labels) - 5

height = -d_ground
T_sensor_2_world = np.array([0, 0, height])
m = np.array([1, 0, 0])
s = np.cross(-n_ground, m)
R_sensor_2_world = np.stack((m, s, -n_ground), axis=0)

## Show results
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

car_positions = np.array([frame['vehicle_pos'][0] for frame in labels])
car_positions = (R_sensor_2_world @ car_positions.T).T + T_sensor_2_world
ax.plot(car_positions[data_frame_begin: data_frame_end, 0], car_positions[data_frame_begin: data_frame_end, 1], car_positions[data_frame_begin: data_frame_end, 2], 'b', linewidth=1.0)

for frame in show_frames:
    if frame < 0 or frame >= len(res):
        break
    ## Draw ground truth
    v = np.array(labels[frame]['keypoints_world_all'][0])[:, 1:4]

    v = (R_sensor_2_world @ v.T).T + T_sensor_2_world

    pos_gt = np.array(labels[frame]['vehicle_pos'][0])
    skeleton = [v[item] for item in car_model.skeleton_24]
    framed = Line3DCollection(skeleton, colors='k', linewidths=0.2, linestyles='-')
    ax.add_collection3d(framed)

    ## Draw the estimated shape
    x_ref = res[frame]['x_ref']
    pos = x_ref[:3]
    theta = x_ref[4:7]
    mu = res[frame]['mu']
    u = mu[:, 3:6]
    base = mu[:, :3]

    R = Rt.from_rotvec(theta).as_matrix()
    u = (R @ u.T).T + pos
    base = (R @ base.T).T + pos

    u = (R_sensor_2_world @ u.T).T + T_sensor_2_world

    faces = ConvexHull(u).simplices
    shape3D = Poly3DCollection(u[faces], alpha=0.5, linewidths=0.1,  edgecolors='k', linestyles=':')
    ax.add_collection3d(shape3D)

    ## Draw radar points
    pts = np.asarray(radar_point[frame])

    pts = (R_sensor_2_world @ pts.T).T + T_sensor_2_world
    
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c='r', marker='o', s=1.0)


# Set coordinate axis range
ax.set_xlim(-8, 8)
ax.set_ylim(20, 70)
ax.set_zlim(-5, 5)
ax.set_box_aspect((16, 50, 10))
ax.elev, ax.azim, ax.roll = 22.621160366335115, -42.91003016999582, 0

ax.grid(True)
# ax.set_xticks([])
# ax.set_yticks([])
ax.set_zticks([])
# plt.axis('off')
plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
plt.savefig('show_track_in_plot.jpg', transparent=True, bbox_inches='tight', dpi=640, pad_inches=0.0)
# plt.show()
# print(ax.elev, ax.azim, ax.roll)
# print(ax.get_xlim(), ax.get_ylim(), ax.get_zlim())