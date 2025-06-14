import cv2
import numpy as np
import pandas as pd
import transforms3d
import glob
import os
import re


# 图像质量检测函数
def assess_image_quality(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var

# 将机械臂末端的姿态向量转换为旋转矩阵和位移向量
def pose_vectors_to_end2base_transforms(pose_vectors):
    R_end2bases = []
    t_end2bases = []
    for pose_vector in pose_vectors:
        R_end2base = euler_to_rotation_matrix(pose_vector[3], pose_vector[4], pose_vector[5])
        t_end2base = pose_vector[:3]  # 位移向量
        R_end2bases.append(R_end2base)
        t_end2bases.append(t_end2base)
    return R_end2bases, t_end2bases

# 欧拉角转换为旋转矩阵
def euler_to_rotation_matrix(rx, ry, rz, unit='deg'):
    if unit == 'deg':
        rx, ry, rz = np.radians([rx, ry, rz])
    Rx = transforms3d.axangles.axangle2mat([1, 0, 0], rx)
    Ry = transforms3d.axangles.axangle2mat([0, 1, 0], ry)
    Rz = transforms3d.axangles.axangle2mat([0, 0, 1], rz)
    return np.dot(Rz, np.dot(Ry, Rx))

def rotation_matrix_to_euler_angles(R):
    """
    将旋转矩阵转换为欧拉角 (roll, pitch, yaw)。
    假设旋转顺序为 ZYX（先绕 Z 轴，再绕 Y 轴，最后绕 X 轴）。

    参数:
    R: 3x3 旋转矩阵

    返回:
    roll, pitch, yaw: 分别为绕 X, Y, Z 轴的欧拉角（单位为弧度）
    """
    assert R.shape == (3, 3), "旋转矩阵必须是 3x3"

    # 计算 pitch (绕 Y 轴的旋转)
    pitch = np.arcsin(-R[2, 0])

    # 判断是否发生万向节锁 (gimbal lock)
    if np.abs(R[2, 0]) < 1:
        roll = np.arctan2(R[2, 1], R[2, 2])  # 计算 roll (绕 X 轴的旋转)
        yaw = np.arctan2(R[1, 0], R[0, 0])   # 计算 yaw (绕 Z 轴的旋转)
    else:
        # 如果发生万向节锁，roll 和 yaw 无法确定（自由度丧失）
        roll = 0
        yaw = np.arctan2(-R[1, 2], R[1, 1])

    return roll, pitch, yaw


# 输入部分
file_path = '/home/yuwei/rst-TN_ws/pose/pose_data.xlsx'
if not os.path.exists(file_path):
    raise FileNotFoundError(f"文件未找到: {file_path}")

pose_df = pd.read_excel(file_path)
pose_vectors = pose_df.values

square_size = 108.0  # 棋盘格每个方格的大小（单位：mm）
pattern_size = (8, 6)
images = glob.glob('/home/yuwei/rst-TN_ws/img/*.jpg')#png也行
# 将图片按照命名尾数字顺序排序
images.sort(key=lambda x: int(re.search(r'(\d+)(?=\.[a-zA-Z]+$)', x).group(0)))

obj_points = []
img_points = []
objp = np.zeros((np.prod(pattern_size), 3), dtype=np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size

det_success_num = 0
filtered_pose_vectors = []
low_quality_images = []
images = images[:95]
# 遍历图像，检测棋盘格角点
for i, image in enumerate(images):
    img = cv2.imread(image)
    if img is None:
        print(f"无法读取图像: {image}")
        continue

    quality_score = assess_image_quality(img)
    quality_threshold = 50
    print(f"图像 {image} 的质量分数: {quality_score}")

    if quality_score < quality_threshold:
        low_quality_images.append(image)
        cv2.putText(img, "Low Quality", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Low Quality Image', img)
        cv2.waitKey(1000)

    elif i == 73 or i == 98:
        pass
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, pattern_size)
        if ret:
            det_success_num += 1
            obj_points.append(objp)
            img_points.append(corners)
            filtered_pose_vectors.append(pose_vectors[i])
            cv2.drawChessboardCorners(img, pattern_size, corners, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)

cv2.destroyAllWindows()

pose_vectors = np.array(filtered_pose_vectors)

# 相机标定
ret, K, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
print("相机内参矩阵 (K):\n", K)
print("畸变系数:\n", dist_coeffs)

R_board2cameras = []
t_board2cameras = []

# 求解标定板在相机坐标系中的位姿
for i in range(det_success_num):
    ret, rvec, t_board2camera = cv2.solvePnP(obj_points[i], img_points[i], K, dist_coeffs)
    R_board2camera, _ = cv2.Rodrigues(rvec)
    R_board2cameras.append(R_board2camera)
    t_board2cameras.append(t_board2camera)

    print(f"图像 {i + 1} 中的标定板在相机坐标系下的旋转矩阵 (R_board2camera):")
    print(R_board2camera)
    print(f"图像 {i + 1} 中的标定板在相机坐标系下的位移向量 (t_board2camera):")
    print(t_board2camera, "（单位：mm）")  # 加入单位说明

R_end2bases, t_end2bases = pose_vectors_to_end2base_transforms(pose_vectors)

assert len(R_end2bases) == len(t_end2bases), "R_end2bases 和 t_end2bases 长度不一致"
assert len(R_board2cameras) == len(t_board2cameras), "R_board2cameras 和 t_board2cameras 长度不一致"
assert len(R_end2bases) == len(R_board2cameras), "机械臂数据与图像姿态数据不匹配"

# 手眼标定，计算相机相对于机械臂末端的位姿
R_camera2end, t_camera2end = cv2.calibrateHandEye(R_end2bases, t_end2bases, R_board2cameras, t_board2cameras,
                                                  method=cv2.CALIB_HAND_EYE_TSAI)

# 构造相机到末端的齐次变换矩阵
T_camera2end = np.eye(4)
T_camera2end[:3, :3] = R_camera2end
T_camera2end[:3, 3] = t_camera2end.reshape(3)

# 设置打印选项
np.set_printoptions(suppress=True, precision=10)  # 不使用科学计数法，保留10位有效数字
roll,pitch,yaw = rotation_matrix_to_euler_angles(R_camera2end)


print("相机到末端的旋转矩阵:")
print(R_camera2end)
print("相机到末端的位移向量:")
print(t_camera2end, "（单位：mm）")  # 加入单位说明
print("相机到末端的齐次变换矩阵:")
print(T_camera2end)
print("相机到末端的欧拉角:")
print("roll:", roll, "pitch:", pitch, "yaw:", yaw)

# 反算标定板在机械臂末端下的位姿，并计算机械臂基座下的标定板位姿
for i in range(det_success_num):
    # 将标定板在相机坐标系下的位姿转换到机械臂末端坐标系
    T_board2camera = np.eye(4)
    T_board2camera[:3, :3] = R_board2cameras[i]
    T_board2camera[:3, 3] = t_board2cameras[i].reshape(3)

    T_board2end = np.dot(T_camera2end, T_board2camera)

    # 计算标定板在机械臂基座下的位姿
    T_end2base = np.eye(4)
    T_end2base[:3, :3] = R_end2bases[i]
    T_end2base[:3, 3] = t_end2bases[i]

    T_board2base = np.dot(T_end2base, T_board2end)

    print(f"图像 {i + 1} 中的标定板在基座坐标系下的齐次变换矩阵 (T_board2base):")
    print(T_board2base)

# 输出质量较差的图像
if low_quality_images:
    print("质量较差的图像:")
    for low_quality_image in low_quality_images:
        print(low_quality_image)
else:
    print("所有图像质量良好。")

# 将结果保存到文件
with open('biaoding_result.txt', 'a') as f:
    f.write("相机到末端的旋转矩阵:\n")
    np.savetxt(f, R_camera2end, fmt='%.10f')
    f.write("\n相机到末端的位移向量:\n")
    np.savetxt(f, t_camera2end, fmt='%.10f')
    f.write("\n相机到末端的齐次变换矩阵:\n")
    np.savetxt(f, T_camera2end, fmt='%.10f')
    f.write("\n相机到末端的欧拉角:\n")
    f.write(f"roll: {roll}, pitch: {pitch}, yaw: {yaw}\n")
    f.write("\n相机内参\n")
    np.savetxt(f, K, fmt='%.10f')
    f.write("\n畸变系数:\n")
    np.savetxt(f, dist_coeffs, fmt='%.10f')


import yaml
import os

# 定义标定参数文件路径
output_path = os.path.expanduser("~/camera_info/kinect.yaml")

# 将标定结果格式化为字典
calibration_data = {
    "camera_name": "kinect",
    "camera_matrix": {
        "rows": 3,
        "cols": 3,
        "data": K.flatten().tolist()
    },
    "distortion_model": "plumb_bob",
    "distortion_coefficients": {
        "rows": 1,
        "cols": len(dist_coeffs),
        "data": dist_coeffs.flatten().tolist()
    },
    "rectification_matrix": {
        "rows": 3,
        "cols": 3,
        "data": np.eye(3).flatten().tolist()  # 假设没有进行校正
    },
    "projection_matrix": {
        "rows": 3,
        "cols": 4,
        "data": np.hstack((K, np.zeros((3, 1)))).flatten().tolist()
    }
}

# 将数据写入 YAML 文件
os.makedirs(os.path.dirname(output_path), exist_ok=True)  # 确保目录存在
with open(output_path, "w") as yaml_file:
    yaml.dump(calibration_data, yaml_file, default_flow_style=False)

print(f"标定参数已保存到 {output_path}")
