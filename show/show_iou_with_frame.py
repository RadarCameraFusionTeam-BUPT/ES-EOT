import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# adjustable parameters
scenario = 'bus_change_lane'
density = 10.0

# Set scenario
root_path = os.path.join(os.path.dirname(__file__), '../')
paths = [
    os.path.join(root_path, 'ES-EOT/metrics_result'), 
    os.path.join(root_path, 'ES-EOT-no-ES/metrics_result'), 
    os.path.join(root_path, 'ES-EOT-CV-model/metrics_result'), 
    os.path.join(root_path, 'ES-EOT-CV-model-no-ES/metrics_result'), 
    os.path.join(root_path, 'GPEOT/metrics_result'), 
    os.path.join(root_path, 'GPEOT-P/metrics_result'), 
    os.path.join(root_path, 'MEM-EKFstar/metrics_result')]

curve_label = ['ES-EOT (CTRV+ES)', 'ES-EOT (CTRV)', 'ES-EOT (CV+ES)', 'ES-EOT (CV)', 'GPEOT', 'GPEOT-P', 'MEM-EKF*']
sub_title = {
    'bus_change_lane': 'Change Lane',
    'turn_around': 'Turn Around',
}

# data range of the vehicle during normal driving conditions
data_frame_begin = 8
data_frame_end = -5

all_res = []

for path in paths:
    res = []
    for file in os.listdir(path):
        file_name_without_extension = os.path.splitext(file)[0]

        file_name_split = file_name_without_extension.split('-')
        dens, idx = file_name_split[1:]
        dens, idx = float(dens), float(idx)
        if file_name_split[0] == scenario and density == dens:
            metric = np.load(os.path.join(path, file), allow_pickle=True).item()
            res.append(metric['iou'][data_frame_begin: data_frame_end])
    all_res.append(res)

all_res = np.asarray(all_res)
frame_len = np.arange(all_res.shape[2])
mean_value = np.mean(all_res, axis=1)
max_value = np.max(all_res, axis=1)
min_value = np.min(all_res, axis=1)

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

for j in range(3):
    axes[j].set_xlabel('frame', fontsize=20)
    axes[j].set_ylabel('IOU', fontsize=20)
    axes[j].set_ylim((0, 1.0))
    axes[j].tick_params(axis='x', labelsize=20)
    axes[j].tick_params(axis='y', labelsize=20)
    for i in range(len(paths)):
        axes[j].plot(frame_len, mean_value[i, :, j], label=curve_label[i])
        # axes[j].fill_between(frame_len, min_value[i, :, j], max_value[i, :, j], alpha=0.3)

axes[0].set_title('XY-plane', fontsize=20)
axes[1].set_title('YZ-plane', fontsize=20)
axes[2].set_title('ZX-plane', fontsize=20)

plt.suptitle('{} Scenario'.format(sub_title[scenario]), fontsize=20,  fontweight='bold')

plt.tight_layout()
font_props = FontProperties(weight='bold', size=16)
plt.legend(bbox_to_anchor=(0.6, -0.2), ncol=4, prop=font_props)
plt.savefig('iou_with_frame.jpg', dpi=640, bbox_inches='tight')
# plt.show()