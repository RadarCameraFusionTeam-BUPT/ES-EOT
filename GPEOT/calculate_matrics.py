import numpy as np
import os, sys
from config import *
import torch as th

from scipy.spatial.transform import Rotation as Rt
from FuncTools import quaternion_to_rotation_matrix, K, quaternion_to_rotation_matrix

if not os.path.join(os.path.dirname(__file__), '../') in sys.path:
    sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import car_model
from metrics import *

# Get ground truth and tracking result
labels = np.load(os.path.join(os.path.dirname(__file__),\
            '../data/turn_around/labels.npy'), allow_pickle=True)
res = np.load(os.path.join(os.path.dirname(__file__),\
                './GPEOT_result.npy'), allow_pickle=True)

u_f = th.tensor(res[0]['u_f'], dtype=th.float64)

theta_f = u_f[:, 0]
phi_f = u_f[:, 1]
x_f = th.cos(phi_f) * th.cos(theta_f)
y_f = th.cos(phi_f) * th.sin(theta_f)
z_f = th.sin(phi_f)
p_f = th.stack((x_f, y_f, z_f), dim = 1)

K_uf_uf = K(u_f, u_f)
H_f = K_uf_uf @ th.inverse(K_uf_uf)

metrics = {'iou': [], 'e_v': []}

for frame in range(len(res)):
    pos = res[frame]['pos']
    quat = th.tensor(res[frame]['quat'], dtype=th.float64)
    f = th.tensor(res[frame]['f'], dtype=th.float64)

    R=quaternion_to_rotation_matrix(quat).numpy()

    len_f = H_f @ f
    extend = p_f * len_f.reshape((-1, 1))
    extend = (R @ extend.numpy().T).T + pos
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