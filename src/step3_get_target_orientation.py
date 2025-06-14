# get orientation of target
import rospy
import tf
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
from cv_bridge import CvBridge
import numpy as np
import cv2
import os
import pandas as pd
from datetime import datetime

def compute_pose_and_euler_angles(O, X, Y):
    """
    计算三维坐标系的姿态（旋转矩阵和平移向量）以及欧拉角。

    参数:
        O (np.ndarray): 原点坐标，形状为 (3,)。
        X (np.ndarray): X 轴上的点坐标，形状为 (3,)。
        Y (np.ndarray): Y 轴上的点坐标，形状为 (3,)。

    返回:
        R (np.ndarray): 旋转矩阵，形状为 (3, 3)。
        t (np.ndarray): 平移向量，形状为 (3,)。
        euler_angles (np.ndarray): 欧拉角 [yaw, pitch, roll]，单位为弧度。
    """
    # 计算基向量
    v_x = X - O  # X 轴方向向量
    v_y = Y - O  # Y 轴方向向量
    
    print("np.dot(v_x, v_y)",np.dot(v_x, v_y))
    v_z = np.cross(v_x, v_y)  # Z 轴方向向量
    
    # 归一化基向量
    u_x = v_x / np.linalg.norm(v_x)
    u_y = v_y / np.linalg.norm(v_y)
    u_z = v_z / np.linalg.norm(v_z)
    print("u_x,u_y,u_z",u_x,u_y,u_z)
    # 构建旋转矩阵
    R = np.column_stack((u_x, u_y, u_z))

    # 平移向量
    t = O

    # 计算欧拉角 (ZYX 顺序)
    pitch = np.arcsin(-R[2, 0])  # 俯仰角
    yaw = np.arctan2(R[1, 0], R[0, 0])  # 偏航角
    roll = np.arctan2(R[2, 1], R[2, 2])  # 滚转角
    euler_angles = np.array([yaw, pitch, roll])
    
    return R, t, euler_angles

if __name__ == '__main__':
    target_points = np.load("/home/glab/Hardware_WS/src/handeye_calib/data/trial2/target_points.npy")
    O = target_points[0]
    X = target_points[1]
    Y = target_points[2]
    R_board2base, t, euler_angles = compute_pose_and_euler_angles(O, X, Y)
    board2camera_matrixs = np.load("/home/glab/Hardware_WS/src/handeye_calib/data/trial2/poses/board2camera_matrix.npy")
    ee2base_matrixs = np.load("/home/glab/Hardware_WS/src/handeye_calib/data/trial2/poses/ee2base_matrix.npy")
    assert board2camera_matrixs.shape[0] == ee2base_matrixs.shape[0],"lenths of board2camera_matrixs and ee2base_matrixs are not equal!!!"

    num_samples = board2camera_matrixs.shape[0]
    euler_sum = np.zeros(3)
    for i in range(num_samples):
        board2camera_matrix = board2camera_matrixs[i]
        ee2base_matrix = ee2base_matrixs[i]
        R_board2camera = board2camera_matrix[:3, :3]
        R_camera2board = np.linalg.inv(R_board2camera)
        R_ee2base = ee2base_matrix[:3, :3]
        R_base2ee = np.linalg.inv(R_ee2base)
        R_camera2ee = R_base2ee @ R_board2base @ R_camera2board
        euler = tf.transformations.euler_from_matrix(R_camera2ee)
        print("euler",euler)
        euler_sum += euler
    euler_mean = euler_sum / num_samples
    print("euler_mean",euler_mean)
    np.save("/home/glab/Hardware_WS/src/handeye_calib/data/trial2/target_points.npy")
    

