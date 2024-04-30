import numpy as np
import matplotlib.pyplot as plt
import os

from numpy.random import multivariate_normal as mvnrnd
from numpy.random import poisson

from FuncTools import *

# generate ground truth
trajectoryGT=np.load(os.path.join(os.path.dirname(__file__),\
                '../trajectoryGT.npy'),allow_pickle=True)
gt_center=np.array([[p['x'],p['y']] for p in trajectoryGT]).T
gt_orient=np.array([p['theta'] for p in trajectoryGT])
gt_length=np.array([[p['l'],p['w']] for p in trajectoryGT]).T
gt_vel=np.array([[p['v']*np.cos(p['theta']),p['v']*np.sin(p['theta'])] for p in trajectoryGT]).T
time_steps=len(trajectoryGT)
time_interval=0.05

# gt_center, gt_rotation, gt_orient, gt_length, gt_vel, time_steps, time_interval = get_ground_truth()
gt = np.vstack((gt_center, gt_orient, gt_length, gt_vel))

# nearly constant velocity model 
H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
Ar = np.array([[1, 0, 0.05, 0], [0, 1, 0, 0.05], [0, 0, 1, 0], [0, 0, 0, 1]])
Ap = np.eye(3)

Ch = np.diag([1/4, 1/4]) # covariance of the multiplicative noise
Cv = np.diag([0.5, 0.5]) # covariance of the measurement noise
Cwr = np.diag([1, 1, 0.5, 0.5]) # covariance of the process noise for the kinematic state
Cwp = np.diag([0.05, 0.001, 0.001]) #covariance of the process noise for the shape parameters

lambda_ = 10 # Nr of measurements is Poisson distributed with mean lambda


## Prior
r = np.array([0, 0, 0, 0]).reshape([4,1])
p = np.array([0, 1, 1]).reshape([3,1])

Cr = np.diag([2, 2, 2, 2])
Cp = np.diag([0.2, 2, 2])

plt.figure()
plt.title('Ground Truth and Estimate')
for t in range(time_steps):
    ## generate measurements
    nk = poisson(lambda_)
    while nk == 0:
        nk = poisson(lambda_)
    print(f'Time step: {t}, {nk} Measurements')
    
    y = np.zeros((2, nk))
    for n in range(nk):
        h = -1 + 2 * np.random.rand(1, 2)
        while np.linalg.norm(h) > 1:
            h = -1 + 2 * np.random.rand(1, 2)

        y[:, n] = gt[0:2, t] + h[0, 0] * gt[3, t] * np.array([np.cos(gt[2, t]), np.sin(gt[2, t])]) + \
                  h[0, 1] * gt[4, t] * np.array([-np.sin(gt[2, t]), np.cos(gt[2, t])]) + mvnrnd([0, 0], Cv, 1)
    
    ## measurement update
    r, p, Cr, Cp = measurement_update(y, H, r, p, Cr, Cp, Ch, Cv)
    
    ## visualize estimate and ground truth for every 3rd scan
    if (t % 3) == 1:
        meas_points, = plt.plot(y[0, :], y[1, :], '.k', linewidth=0.5)
        gt_plot, = plot_extent(gt[:, t], '-', 'k', 1)
        est_plot, = plot_extent(np.vstack((r[0:2], p)).reshape([-1]), '-', 'r', 1)
        plt.axis('equal')
        plt.pause(0.1)
    
    ## time update
    r, p, Cr, Cp = time_update(r, p, Cr, Cp, Ar, Ap, Cwr, Cwp)

# add legend with labels
plt.legend([gt_plot, est_plot, meas_points], ['Ground truth', 'Estimate', 'Measurement'], loc='upper left')

# show plot
plt.show()