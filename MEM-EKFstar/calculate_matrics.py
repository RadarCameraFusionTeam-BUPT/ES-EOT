import numpy as np
import os, sys

from scipy.spatial.transform import Rotation as Rt

if not os.path.join(os.path.dirname(__file__), '../') in sys.path:
    sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import car_model
from metrics import *

# Get ground truth and tracking result
labels = np.load(os.path.join(os.path.dirname(__file__),\
            '../data/turn_around/labels.npy'), allow_pickle=True)
res = np.load(os.path.join(os.path.dirname(__file__),\
                './MEM-EKFstar_result.npy'), allow_pickle=True)

metrics = {'iou': [], 'e_v': []}

u = np.linspace(0, 2 * np.pi, 20)
v = np.linspace(0, np.pi, 20)

for frame in range(len(res)):
    r, Cr, p, Cp = res[frame]['r'], res[frame]['Cr'], res[frame]['p'], res[frame]['Cp']

    # detected shape ('u' for real shape, 'base' for radar detections)
    R = Rt.from_euler('xyz',p[:3,0].reshape(3)).as_matrix()
    x_radius, y_radius, z_radius = p[3:,0]
    center=r[:3,0]

    x = x_radius * np.outer(np.cos(u), np.sin(v))
    y = y_radius * np.outer(np.sin(u), np.sin(v))
    z = z_radius * np.outer(np.ones_like(u), np.cos(v))

    points = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
    points_rotated = (R @ points.T).T + r[:3].squeeze()
    ve = np.linalg.norm(r[3:])

    # ground truth shape
    verts = np.array(labels[frame]['keypoints_world_all'][0])[:, 1:4]
    vel = np.linalg.norm(np.array(labels[frame]['velocity'][0]))

    gt_quats = np.array(labels[frame]['vehicle_quats'][0])
    R_gt = Rt.from_quat(gt_quats).as_matrix()
    gt_pos = np.array(labels[frame]['vehicle_pos'][0])

    # iou = iou_of_convex_hulls(points_rotated, verts)
    iou = iou_of_convex_hulls((R_gt.T @ (points_rotated - gt_pos).T).T, (R_gt.T @ (verts - gt_pos).T).T)
    diff_v = difference_between_velocity(ve, vel)

    metrics['iou'].append(iou)
    metrics['e_v'].append(diff_v)

print(np.mean(metrics['iou'], axis=0))
print(np.mean(np.abs(metrics['e_v'])))