import torch as th
import numpy as np
import random
from torch.autograd.functional import jacobian
import time
from config import *

def spherical_angles_3d(theta_points, phi_points):
    # Generate the spherical angles
    theta = np.linspace(
        np.pi / theta_points, 
        2 * np.pi - np.pi / theta_points,
        theta_points, endpoint = True
    )
    phi = np.linspace(
        -np.pi / 2 + np.pi / (2 * phi_points),
        np.pi / 2 - np.pi / (2 * phi_points), 
        phi_points, endpoint = True
    )
    theta = th.tensor(theta, requires_grad = False, dtype = th.float64)
    phi = th.tensor(phi, requires_grad = False, dtype = th.float64)
    # Generate the 3D grid of spherical angles
    theta, phi = th.meshgrid(theta, phi, indexing='ij')
    theta, phi = th.flatten(theta), th.flatten(phi)

    return th.stack((theta, phi), dim = 1)

def generate_sphere_measurements(num_measurements, radius, R):
    """
    Generate random measurements on the surface of a sphere.

    Args:
    num_measurements: integer, representing the number of measurements to generate
    radius: float, representing the radius of the sphere
    R: numpy array of shape (3, 3), representing the covariance of the generated measurements.
    The covariance of the generated measurements is given by R.
    Note that the covariance of the generated measurements is not necessarily symmetric.

    Returns:
    measurements: numpy array of shape (num_measurements, 3), representing the generated measurements
    """

    # generate random points on the unit sphere
    theta = np.random.uniform(0.01, 2.0 * np.pi - 0.01, size=num_measurements)
    phi = np.random.uniform(0.01, np.pi - 0.01, size=num_measurements)
    x = 0 + radius * np.sin(phi) * np.cos(theta)
    y = 0 + radius * np.sin(phi) * np.sin(theta)
    z = 0 + radius * np.cos(phi)

    # stack the coordinates into a (num_measurements, 3) array
    noise = np.random.multivariate_normal([0, 0, 0], R, num_measurements)
    measurements = np.stack((x, y, z), axis=1) + noise

    return measurements

def K(u1, u2):
    u1 = u1.unsqueeze(0) if u1.ndim == 1 else u1 
    u2 = u2.unsqueeze(0) if u2.ndim == 1 else u2

    cos_theta = th.cos(u1[:, 0])
    sin_theta = th.sin(u1[:, 0])
    cos_phi = th.cos(u1[:, 1])
    sin_phi = th.sin(u1[:, 1])

    cos_theta_ = th.cos(u2[:, 0])
    sin_theta_ = th.sin(u2[:, 0])
    cos_phi_ = th.cos(u2[:, 1])
    sin_phi_ = th.sin(u2[:, 1])

    ret = (cos_phi * cos_theta).reshape((-1, 1)) @ (cos_phi_ * cos_theta_).reshape((1, -1)) + \
        (cos_phi * sin_theta).reshape((-1, 1)) @ (cos_phi_ * sin_theta_).reshape((1, -1)) + \
        sin_phi.reshape((-1, 1)) @ sin_phi_.reshape((1, -1))
    ret = th.arccos(th.clamp(ret, -1, 1))
    ret = sigma_f**2 * th.exp(-ret**2 / (2 * l**2)) + sigma_r**2
    return ret.to(th.float64)

def quaternion_to_rotation_matrix(q_k):
    # Check if the norm of the quaternion is equal to 1 within a tolerance
    if not th.allclose(th.linalg.norm(q_k), th.tensor(1.0, dtype = th.float64), rtol=0, atol=1e-8):
        raise AssertionError('Quaternion should be normalized')
    x, y, z, w = q_k
    Rot_k = th.zeros((3, 3), dtype = th.float64)
    Rot_k[0, 0] = 1 - 2*(y**2 + z**2)
    Rot_k[0, 1] = 2*(x*y - w*z)
    Rot_k[0, 2] = 2*(x*z + w*y)
    Rot_k[1, 0] = 2*(x*y + w*z)
    Rot_k[1, 1] = 1 - 2*(x**2 + z**2)
    Rot_k[1, 2] = 2*(y*z - w*x)
    Rot_k[2, 0] = 2*(x*z - w*y)
    Rot_k[2, 1] = 2*(y*z + w*x)
    Rot_k[2, 2] = 1 - 2*(x**2 + y**2)
    return Rot_k

def h(x_k, q_k, u_f, m_k, K_uf_uf_inv):
    m_k = m_k.unsqueeze(0) if m_k.ndim == 1 else m_k 

    f_k = x_k[12:]
    c_k = x_k[:3]
    a = x_k[6:9]

    diffPos = m_k - c_k
    diffPos_norm = th.norm(diffPos, dim = 1)
    diffPos_norm = th.max(diffPos_norm, th.tensor(1e-6, dtype = th.float64))
    p_k = diffPos / diffPos_norm.reshape((-1, 1))
    
    quat = quat_mul(th.cat((a, th.tensor([2.]))) / th.sqrt(4 + th.norm(a)**2), q_k)

    R_L_G = quaternion_to_rotation_matrix(quat).T
    m_L_k = (R_L_G @ p_k.T).T
    x_L, y_L, z_L = m_L_k[:, 0], m_L_k[:, 1], m_L_k[:, 2]

    theta_kl = th.arctan2(y_L, x_L)
    phi_kl = th.arctan2(z_L, th.sqrt(x_L**2 + y_L**2))

    u_k = th.cat((theta_kl.reshape((-1, 1)), phi_kl.reshape((-1, 1))), dim = 1)
    return (
        c_k - m_k + p_k * (K(u_k, u_f) @ K_uf_uf_inv @ f_k).reshape((-1, 1))
    ).flatten()

def skew_symmetric_matrix(v):
    ret = th.zeros((3, 3), dtype = th.float64)
    ret[0, 1] = -v[2]
    ret[0, 2] = v[1]
    ret[1, 0] = v[2]
    ret[1, 2] = -v[0]
    ret[2, 0] = -v[1]
    ret[2, 1] = v[0]
    return ret

def quat_mul(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    ret = th.zeros(4, dtype = th.float64)
    ret[0] = x
    ret[1] = y
    ret[2] = z
    ret[3] = w
    return ret

def exp_from_vector_skew(v):
    fa = th.norm(v)
    if th.abs(fa) < 1e-8:
        return th.eye(3)
    else:
        u = v / fa
        skew_u = skew_symmetric_matrix(u)
        return th.eye(3) + th.sin(fa) * skew_u + (1 - th.cos(fa)) * (skew_u @ skew_u)

def get_Fr_Qr(omega, T, sigma_alpha):
    exp_Fr = exp_from_vector_skew(T / 2 * (-omega))
    F_r = th.cat((
        th.cat((exp_Fr, T * exp_Fr), dim = 1), th.cat((th.zeros((3, 3)), th.eye(3)), dim = 1)
    ))
    omega_norm = th.norm(omega)
    if th.abs(omega_norm) < 1e-8:
        Int_1 = T * th.eye(3)
        Int_2 = T**2 / 2 * th.eye(3)
    else:
        skew_omega = skew_symmetric_matrix(-omega)
        Int_1 = T * th.eye(3) + \
            2 * (1 - th.cos(T / 2 * omega_norm)) / omega_norm**2 * skew_omega + \
            (T - 2 / omega_norm * th.sin(T / 2 * omega_norm)) / omega_norm**2 * (skew_omega @ skew_omega)
        Int_2 = T**2 / 2 * th.eye(3) + \
            1 / omega_norm**2 * (4 / omega_norm * th.sin(T / 2 * omega_norm) - 2 * T * th.cos(T / 2 * omega_norm)) * skew_omega + \
            1 / omega_norm**2 * (T**2 / 2 - 2 * T / omega_norm * th.sin(T / 2 * omega_norm) - 4 / omega_norm**2 * (th.cos(T / 2 * omega_norm) - 1)) * (skew_omega @ skew_omega)

    B = th.cat((th.zeros((3, 3), dtype = th.float64), th.eye(3)))
    G_k = th.cat((
        th.cat((Int_1, Int_2), dim = 1), th.cat((th.zeros((3, 3), dtype = th.float64), T * th.eye(3)), dim = 1)
    )) @ B
    Sigma_alpha = th.eye(3, dtype = th.float64) * sigma_alpha**2
    Q_r = G_k @ Sigma_alpha @ G_k.T
    return F_r, Q_r

def predict(x_k, P_k, q_k, F_t, Q_t, T, K_uf_uf):
    omega = x_k[9:12]
    
    F_r, Q_r = get_Fr_Qr(omega, T, sigma_alpha)
    
    F_f = th.eye(len(x_k) - 12)
    F_k = th.block_diag(F_t, F_r, F_f)

    P_f = P_k[12:, 12:]
    Q_f = (1 / lamb - 1) * P_f
    Q_k = th.block_diag(Q_t, Q_r, Q_f)

    x_k = F_k @ x_k
    P_k = F_k @ P_k @ F_k.T + Q_k
    return x_k, P_k, q_k

def get_R_k(x_k, q_k, u_f, m_k, K_uf_uf_inv):
    m_k = m_k.unsqueeze(0) if m_k.ndim == 1 else m_k 

    f_k = x_k[12:]
    c_k = x_k[:3]
    a = x_k[6:9]

    diffPos = m_k - c_k
    diffPos_norm = th.norm(diffPos, dim = 1)
    diffPos_norm = th.max(diffPos_norm, th.tensor(1e-6, dtype = th.float64))
    p_k = diffPos / diffPos_norm.reshape((-1, 1))
    
    quat = quat_mul(th.cat((a, th.tensor([2.]))) / th.sqrt(4 + th.norm(a)**2), q_k)

    R_L_G = quaternion_to_rotation_matrix(quat).T
    m_L_k = (R_L_G @ p_k.T).T
    x_L, y_L, z_L = m_L_k[:, 0], m_L_k[:, 1], m_L_k[:, 2]

    theta_kl = th.arctan2(y_L, x_L)
    phi_kl = th.arctan2(z_L, th.sqrt(x_L**2 + y_L**2))

    u_k = th.cat((theta_kl.reshape((-1, 1)), phi_kl.reshape((-1, 1))), dim = 1)

    K_uk_uf = K(u_k, u_f)
    R_f = K(u_k, u_k) - K_uk_uf @ K_uf_uf_inv @ K_uk_uf.T

    R_k = th.block_diag(
        *(
            th.diag(R_f).reshape((-1, 1, 1)) * (p_k.reshape((-1, 3, 1)) @ p_k.reshape((-1, 1, 3)))
        )
    ) + th.kron(th.eye(len(R_f)), R_line)
    return R_k

def update(x_k, P_k, m_k, q_k, u_f, K_uf_uf_inv):
    a = x_k[6:9]

    H_k = jacobian(h, (x_k, q_k, u_f, m_k, K_uf_uf_inv))[0]
    h_k = h(x_k, q_k, u_f, m_k, K_uf_uf_inv)

    R_k = get_R_k(x_k, q_k, u_f, m_k, K_uf_uf_inv)

    S_k = H_k @ P_k @ H_k.T + R_k
    K_k = P_k @ H_k.T @ th.linalg.inv(S_k)

    # update the state estimate
    x_k = x_k + K_k @ (-h_k)

    # update the error covariance matrix
    I_KH = th.eye(len(x_k)) - K_k @ H_k
    P_k = I_KH @ P_k @ I_KH.T + K_k @ R_k @ K_k.T
    P_k = (P_k + P_k.T) / 2

    # update error state
    a = x_k[6:9]
    q_k = quat_mul(th.cat((a, th.tensor([2.]))) / th.sqrt(4 + th.norm(a)**2), q_k)
    q_k = q_k / th.norm(q_k)
    x_k[6:9] = th.zeros(3)

    return x_k, P_k, q_k
    
    
if __name__ == '__main__':
    # print(spherical_angles_3d(2, 3))
    pass