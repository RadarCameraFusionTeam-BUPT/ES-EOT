import numpy as np
import os
from config import *

from scipy.spatial.transform import Rotation as Rt

if not os.path.join(os.path.dirname(__file__), '../') in sys.path:
    sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import car_model
from metrics import *

# Get ground truth and tracking result
labels = np.load(os.path.join(os.path.dirname(__file__),\
            '../data/turn_around/labels.npy'), allow_pickle=True)
res = np.load(os.path.join(os.path.dirname(__file__),\
                './ES-EOT_result.npy'), allow_pickle=True)

metrics = {'iou': [], 'e_v': []}

for frame in range(len(res)):
    x_ref = res[frame]['x_ref']
    pos = x_ref[:3]
    theta = x_ref[4:7]
    mu = res[frame]['mu']
    u = mu[:, 3:6]
    base = mu[:, :3]
    m = res[frame]['m']

    # detected shape ('u' for real shape, 'base' for radar detections)
    R = Rt.from_rotvec(theta).as_matrix()
    u = (R @ u.T).T + pos
    base = (R @ base.T).T + pos
    v = x_ref[3]

    # ground truth shape
    verts = np.array(labels[frame]['keypoints_world_all'][0])[:, 1:4]
    vel = np.linalg.norm(np.array(labels[frame]['velocity'][0]))
    gt_quats = np.array(labels[frame]['vehicle_quats'][0])
    R_gt = Rt.from_quat(gt_quats).as_matrix()
    gt_pos = np.array(labels[frame]['vehicle_pos'][0])

    # iou = iou_of_convex_hulls(u, verts)
    iou = iou_of_convex_hulls((R_gt.T @ (u - gt_pos).T).T, (R_gt.T @ (verts - gt_pos).T).T)
    diff_v = difference_between_velocity(v, vel)

    metrics['iou'].append(iou)
    metrics['e_v'].append(diff_v)

print(np.mean(metrics['iou'], axis=0))
print(np.mean(np.abs(metrics['e_v'])))