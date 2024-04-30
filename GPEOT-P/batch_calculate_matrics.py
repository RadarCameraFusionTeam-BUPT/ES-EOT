import numpy as np
import os, sys
from config import *
import torch as th
from tqdm import tqdm

from scipy.spatial.transform import Rotation as Rt
from FuncTools import quaternion_to_rotation_matrix, K, quaternion_to_rotation_matrix, reconstruct_from_projections

if not os.path.join(os.path.dirname(__file__), '../') in sys.path:
    sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import car_model
from metrics import *

scenario = 'turn_around'

# Get ground truth and tracking result
labels = np.load(os.path.join(os.path.dirname(__file__),\
            '../data/{}/labels.npy'.format(scenario)), allow_pickle=True)

res_dir_path = os.path.join(os.path.dirname(__file__), 'res')

# Output metrics path
metric_dir_path = os.path.join(os.path.dirname(__file__), 'metrics_result')
if not os.path.exists(metric_dir_path):
    os.makedirs(metric_dir_path)

res_file_list = os.listdir(res_dir_path)
res_file_list = [file for file in res_file_list if file.split('-')[0] == scenario]

for res_file in tqdm(res_file_list, desc='Processing files', ncols=100):
    file_path = os.path.join(res_dir_path, res_file)
    res = np.load(file_path, allow_pickle=True)

    metrics = {'iou': [], 'e_v': []}

    file_name_without_ext = os.path.splitext(res_file)[0]

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

    metrics['iou'] = np.asarray(metrics['iou'])
    metrics['e_v'] = np.asarray(metrics['e_v'])
    
    np.save(os.path.join(metric_dir_path, '{}.npy'.format(file_name_without_ext)), metrics, allow_pickle=True)