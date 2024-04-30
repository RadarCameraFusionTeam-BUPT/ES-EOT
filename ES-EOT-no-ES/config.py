import numpy as np
import os, sys
from scipy.linalg import block_diag

if not os.path.join(os.path.dirname(__file__), '../') in sys.path:
    sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import car_model

# Set intrinsic parameters of the camera
K = np.array([
    [3.32553755e+03, 0.00000000e+00, 1.92000000e+03],
    [0.00000000e+00, 3.32553755e+03, 1.08000000e+03],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
])

# If the angular velocity is less than eps
# it is considered 0
eps = 1e-6

# elastic coefficient
epsilon = 0

# damping factor
rho = 20

# time interval
dt = 0.05

# Process noise
W = block_diag(np.zeros((3, 3)), np.eye(1)*0.01, np.zeros((3, 3)), np.eye(3) * 1, np.eye(2)*0.1)
W_vartheta = np.eye(9)

# Skeleton
control_points = car_model.keypoints.copy()

# Number of reflector points
N_T = 24

# Ground point idx
skeleton_knots_id = np.arange(N_T)
ground_id = np.array([24, 25, 26, 27])
corner_id = np.array([24, 25, 26, 27])

# relationship of keypoints and extend
keypoint_id_to_extend = {
    24: np.array([0.5, -0.5]),
    25: np.array([-0.5, -0.5]),
    26: np.array([-0.5, 0.5]),
    27: np.array([0.5, 0.5])
}

# Measurement noise
Q = np.eye(3) * 0.5
Q_inv = np.linalg.inv(Q)

Q_rot = np.array([[0.1]])
Q_rot_inv = np.linalg.inv(Q_rot)

Q_sym = np.eye(3) * 0.05
Q_sym_inv = np.linalg.inv(Q_sym)

V_c = np.eye(2) * 5
V_c_inv = np.linalg.inv(V_c)

Q_ground = np.array([[1e-4]])
Q_ground_inv = np.linalg.inv(Q_ground)

# ground vector and d
n_ground = np.array([-0.002671761716973599, 0.9396342322901945, 0.3421700910333109])
d_ground = -6.73240454224306

# Flip
D = np.diag([1.0, -1.0, 1.0])
flip_id = np.array([1, 0, 3, 2, 5, 4, 16, 19, 18, 17, 11, 10, 13, 12, 15, 14, 6, 9, 8, 7, 21, 20, 23, 22])

# Car heading direction
u_d = np.array([1.0, 0.0, 0.0])

# Number of iterations of VB
N_iter = 3

# Constant parameters
H_r = np.zeros((3, 12))
H_r[:, :3] = np.eye(3)
H_theta = np.zeros((3, 12))
H_theta[:, 4:7] = np.eye(3)
H_omega = np.zeros((3, 12))
H_omega[:, 7:10] = np.eye(3)

H_u = np.zeros((3, 9))
H_u[:, :3] = np.eye(3)
H_varpi = np.zeros((3, 9))
H_varpi[:, 3:6] = np.eye(3)
decay = 0.3

class State:
    def __init__(self, x_ref, dx, P, m, mu, Sigma):
        self.x_ref = x_ref
        self.dx = dx
        self.P = P
        self.m = m
        self.mu = mu
        self.Sigma = Sigma

    def __str__(self):
        return "x_ref: \n" + str(self.x_ref) + "\ndx: \n" + str(self.dx) + "\th: \n" + str(self.P) + "\nm: \n" + str(self.m) + "\nmu: \n" + str(self.mu) + "\nSigma: \n" + str(self.Sigma)

    def copy(self):
        new_obj = State(self.x_ref.copy(), self.dx.copy(), self.P.copy(), self.m.copy(), self.mu.copy(), self.Sigma.copy())
        return new_obj