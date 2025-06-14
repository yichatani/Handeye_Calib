import pandas as pd
import numpy as np
import os
import tf

# 设置文件路径
pose_dir = "/home/glab/Hardware_WS/src/handeye_calib/data/trial1/poses"
input_file = os.path.join(pose_dir, "ee2base_pose_back.xlsx")
output_file = os.path.join(pose_dir, "ee2base_pose.xlsx")

# 读取 Excel 文件
pose_data = pd.read_excel(input_file)

# 验证列名是否存在
required_columns = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']
if not all(col in pose_data.columns for col in required_columns):
    raise ValueError(f"Excel file must contain columns: {required_columns}")

# 处理 x, y, z 列：除以 1000
pose_data['x'] = pose_data['x'] / 1000
pose_data['y'] = pose_data['y'] / 1000
pose_data['z'] = pose_data['z'] / 1000

# 处理 roll, pitch, yaw 列：角度制转为弧度制
pose_data['roll'] = np.deg2rad(pose_data['roll'])
pose_data['pitch'] = np.deg2rad(pose_data['pitch'])
pose_data['yaw'] = np.deg2rad(pose_data['yaw'])

# 保存处理后的数据到新文件
pose_data.to_excel(output_file, index=False)
print(f"Processed data saved to {output_file}")

pose_data_nps = pose_data.to_numpy()
# turn xyzrpy to T
pose_data_matrixs = []
for pose_data_np in pose_data_nps:
    pose_data_matrix = np.eye(4)
    pose_data_matrix[:3, :3] = tf.transformations.euler_matrix(pose_data_np[3], pose_data_np[4], pose_data_np[5])[:3, :3]
    pose_data_matrix[:3, 3] = pose_data_np[:3]
    pose_data_matrixs.append(pose_data_matrix.copy())
pose_data_matrixs = np.array(pose_data_matrixs)
np.save(pose_dir+"/ee2base_matrix.npy", pose_data_matrixs)

