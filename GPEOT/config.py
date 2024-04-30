import torch as th

# number of basis points 
theta_points = 10
phi_points = 10

# Time interval
T = 0.05

# Parameters of Gaussian process
sigma_f = th.tensor(1.0, dtype=th.float64)
sigma_r = th.tensor(0.5, dtype=th.float64)
l = th.tensor(th.pi / 8, dtype=th.float64)

# Process noise
sigma_c = th.tensor(1.0, dtype=th.float64)
sigma_alpha = th.tensor(1.0, dtype=th.float64)
lamb = th.tensor(0.99, dtype=th.float64)

# Measurement noise
R_line = (0.5) * th.eye(3, dtype=th.float64)