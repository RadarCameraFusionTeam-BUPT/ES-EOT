from FuncTools import *
import torch as th
import time
import os
from config import *
from scipy.spatial.transform import Rotation as Rt
import numpy as np

# Load measurement data
radar_point = np.load(os.path.join(os.path.dirname(__file__),\
            '../data/turn_around/radar/10.0-0.npy'), allow_pickle=True)
labels = np.load(os.path.join(os.path.dirname(__file__),\
            '../data/turn_around/labels.npy'), allow_pickle=True)

pos0 = labels[0]['vehicle_pos'][0]
vel_0 = labels[0]['velocity'][0]
q_0 = labels[0]['vehicle_quats'][0]

# Basis points
basis_points = polar_angles(theta_points)
u_f = th.stack((basis_points, basis_points, basis_points))

# Prior
## Kinematic state
x_t = th.tensor([pos0[0], pos0[1], pos0[2], vel_0[0], vel_0[1], vel_0[2]], dtype=th.float64)
P_t = 0.1 * th.eye(len(x_t), dtype=th.float64)

## Rotation state
quat = th.tensor(q_0, dtype=th.float64)
a, omega = th.tensor([0., 0., 0.], dtype=th.float64), th.tensor([0., 0., 0.], dtype=th.float64)
x_r = th.cat((a, omega))
P_r = th.eye(len(x_r), dtype=th.float64)

## Extend state
K_uf_uf = K(u_f, u_f) + 1e-6 * th.eye(theta_points, dtype=th.float64)
K_uf_uf_inv = th.inverse(K_uf_uf)

f = th.zeros(3 * theta_points, dtype=th.float64)
P_f = th.block_diag(K_uf_uf[0], K_uf_uf[1], K_uf_uf[2])

## Cat all states
x_k = th.cat((x_t, x_r, f))
P_k = th.block_diag(P_t, P_r, P_f)

# Process model
F_t = th.kron(th.tensor([[1, T], [0, 1]], dtype=th.float64), th.eye(3))
Q_t = th.kron(th.tensor([[T**3 / 3, T**2 / 2],[T**2 / 2, T]], dtype=th.float64), (sigma_c**2) * th.eye(3))

# Set print properties
th.set_printoptions(2, sci_mode = False)

res = []

for i in range(len(radar_point)):
    y = np.array(radar_point[i])
    m_k = th.tensor(y, dtype = th.float64)

    ## Sphere surface data
    # m_k = th.tensor(generate_sphere_measurements(10, 1.0, 0.5 * np.eye(3)), dtype = th.float64)
    
    x_k, P_k, quat = predict(x_k, P_k, quat, F_t, Q_t, T, K_uf_uf)
    x_k, P_k, quat = update(x_k, P_k, m_k, quat, u_f, K_uf_uf_inv)
    
    now = dict()
    now['pos'] = x_k[0:3].detach().numpy()
    now['vel'] = x_k[3:6].detach().numpy()
    now['quat'] = quat.detach().numpy()
    now['f'] = x_k[12:].detach().numpy()
    
    now['P_pos'] = P_k[0:3, 0:3].detach().numpy()
    now['P_vel'] = P_k[3:6, 3:6].detach().numpy()
    now['P_quat'] = P_k[6:9, 6:9].detach().numpy()
    now['P_f'] = P_k[12:, 12:].detach().numpy()

    now['u_f'] = u_f.detach().numpy()
    res.append(now)

    print('frame', i)
    print(x_k)

np.save(os.path.join(os.path.dirname(__file__),\
    'GPEOT-P_result.npy'), res, allow_pickle = True)