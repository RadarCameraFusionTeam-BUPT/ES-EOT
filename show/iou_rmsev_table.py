import numpy as np
import os

# adjustable parameters
scenario = 'turn_around'

# set path
root_path = os.path.join(os.path.dirname(__file__), '../')
paths = [
    os.path.join(root_path, 'ES-EOT-CV-model/metrics_result'), 
    os.path.join(root_path, 'ES-EOT-CV-model-no-ES/metrics_result'), 
    os.path.join(root_path, 'ES-EOT/metrics_result'), 
    os.path.join(root_path, 'ES-EOT-no-ES/metrics_result'), 
    os.path.join(root_path, 'GPEOT/metrics_result'), 
    os.path.join(root_path, 'GPEOT-P/metrics_result'), 
    os.path.join(root_path, 'MEM-EKFstar/metrics_result')]
methods = ['ES-EOT (CV model)', 'ES-EOT (CV no ES)', 'ES-EOT (CTRV model)', 'ES-EOT (CTRV no ES)', 'GPEOT', 'GPEOT-P', 'MEM-EKF*']
density = [1.0, 5.0, 10.0]

# data range of the vehicle during normal driving conditions
data_frame_begin = 8
data_frame_end = -5

all_res = []

for path in paths:
    dens_res = []
    for dens in density:
        res = []
        for file in os.listdir(path):
            file_name_without_extension = os.path.splitext(file)[0]

            file_name_split = file_name_without_extension.split('-')
            den, idx = file_name_split[1:]
            den, idx = float(den), float(idx)
            if file_name_split[0] == scenario and den == dens:
                metric = np.load(os.path.join(path, file), allow_pickle=True).item()
                e_v = metric['e_v'][data_frame_begin: data_frame_end]
                mac_e_v = np.square(e_v)
                res.append(metric['iou'][data_frame_begin: data_frame_end])
                res[-1] = np.hstack((res[-1], mac_e_v.reshape([-1, 1])))
        res = np.mean(res, axis=(0, 1))
        res[-1] = np.sqrt(res[-1])
        dens_res.append(res)
    all_res.append(dens_res)

all_res = np.asarray(all_res)
mean_value = all_res.transpose((0, 2, 1))

for name, val in zip(methods, mean_value):
    val_str = ["{:.3f}".format(num) for num in val.flatten()]
    print(name, *val_str, sep=' & ')
