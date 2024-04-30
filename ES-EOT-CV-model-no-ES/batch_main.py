from config import *
from FuncTools import *
import numpy as np
import time
import os
from tqdm import tqdm

# adjustable parameters
scenario = 'turn_around'

# Set path
data_root_path = os.path.join(os.path.dirname(__file__),\
            '../data/{}/'.format(scenario))
label_path = os.path.join(data_root_path, 'labels.npy')
radar_dir_path = os.path.join(data_root_path, 'radar')
keypoints_det_path = os.path.join(data_root_path, 'vision/output-keypoints.npy')

res_dir_path = os.path.join(os.path.dirname(__file__), 'res')
if not os.path.exists(res_dir_path):
    os.makedirs(res_dir_path)

# Load measurement data
labels = np.load(label_path, allow_pickle=True)
keypoints_det = np.load(keypoints_det_path, allow_pickle=True)

radar_file_list = os.listdir(radar_dir_path)

for radar_file in tqdm(radar_file_list, desc='Processing files', ncols=100):
    file_path = os.path.join(radar_dir_path, radar_file)
    radar_point = np.load(file_path, allow_pickle=True)

    file_name_without_ext = os.path.splitext(radar_file)[0]

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

        Theta = update(Theta, z_r, z_c)

        now = dict()
        now['x_ref'] = Theta.x_ref.copy()
        now['P'] = Theta.P.copy()
        now['m'] = Theta.m.copy()
        now['mu'] = Theta.mu.copy()
        now['Sigma'] = Theta.Sigma.copy()
        res.append(now)

        Theta = predict(Theta)
        

    np.save(os.path.join(res_dir_path,\
         '{}-{}.npy'.format(scenario, file_name_without_ext)), res, allow_pickle = True)