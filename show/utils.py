import numpy as np
import os, sys

if not os.path.join(os.path.dirname(__file__), '../') in sys.path:
    sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import car_model

# Set intrinsic parameters of the camera
K = np.array([
    [3.32553755e+03, 0.00000000e+00, 1.92000000e+03],
    [0.00000000e+00, 3.32553755e+03, 1.08000000e+03],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
])

# ground vector and d
n_ground = np.array([-0.002671761716973599, 0.9396342322901945, 0.3421700910333109])
d_ground = -6.73240454224306

# corner point idx
corner_id = np.array([24, 25, 26, 27])
skeleton_knots_id = np.arange(24)