import numpy as np
import os, sys
from config import *
import torch as th

from scipy.spatial.transform import Rotation as Rt
from FuncTools import quaternion_to_rotation_matrix, K, quaternion_to_rotation_matrix, reconstruct_from_projections

if not os.path.join(os.path.dirname(__file__), '../') in sys.path:
    sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import car_model
from metrics import *

# Get ground truth and tracking result
labels = np.load(os.path.join(os.path.dirname(__file__),\
            '../data/turn_around/labels.npy'), allow_pickle=True)
res = np.load(os.path.join(os.path.dirname(__file__),\
                './GPEOT-P_result.npy'), allow_pickle=True)

num_basis_points = len(res[0]['u_f'])

u_f = th.tensor(res[0]['u_f'], dtype=th.float64)

metrics = {'iou': [], 'e_v': []}

for frame in range(len(res)):
    pos = th.tensor(res[frame]['pos'], dtype=th.float64)
    quat = th.tensor(res[frame]['quat'], dtype=th.float64)
    f = th.tensor(res[frame]['f'], dtype=th.float64).reshape(3, theta_points)

    extend = reconstruct_from_projections(pos, quat, f, u_f).numpy()
    v = np.linalg.norm(res[frame]['vel'])

    # ground truth shape
    verts = np.array(labels[frame]['keypoints_world_all'][0])[:, 1:4]
    vel = np.linalg.norm(np.array(labels[frame]['velocity'][0]))

    gt_quats = np.array(labels[frame]['vehicle_quats'][0])
    R_gt = Rt.from_quat(gt_quats).as_matrix()
    gt_pos = np.array(labels[frame]['vehicle_pos'][0])

    # iou = iou_of_convex_hulls(extend, verts)
    iou = iou_of_convex_hulls((R_gt.T @ (extend - gt_pos).T).T, (R_gt.T @ (verts - gt_pos).T).T)
    diff_v = difference_between_velocity(v, vel)

    metrics['iou'].append(iou)
    metrics['e_v'].append(diff_v)

print(np.mean(metrics['iou'], axis=0))
print(np.mean(np.abs(metrics['e_v'])))