import torch as th

# number of basis points 
theta_points = 10

# Number of test layers (the contours will be repeated for this many times)
numLayers = 30

# Time interval
T = 0.05

# Parameters of Gaussian process
sigma_f = th.tensor(1.0, dtype=th.float64)
sigma_r = th.tensor(0.2, dtype=th.float64)
l = th.tensor(th.pi / 8, dtype=th.float64)
mu_s = th.tensor(5 / 6, dtype=th.float64)
sigma_s = th.sqrt(th.tensor(1 / 18, dtype=th.float64))

Proj = th.tensor([[[1, 0, 0], [0, 1, 0]], 
                [[1, 0, 0], [0, 0, 1]], 
                [[0, 1, 0], [0, 0, 1]]], dtype=th.float64)

# Process noise
sigma_c = th.tensor(1.0, dtype=th.float64)
sigma_alpha = th.tensor(1.0, dtype=th.float64)
lamb = th.tensor(0.99, dtype=th.float64)

# Measurement noise
R_line = (0.5) * th.eye(2, dtype=th.float64)