#!/usr/bin/env python
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import tf2_ros
import math
from tf.transformations import quaternion_from_euler,euler_from_quaternion,euler_from_matrix

def quaternion_to_rotation_matrix(q):
    """
    将四元数转换为旋转矩阵。

    参数:
        q (np.ndarray): 四元数，形状为 (4,)，顺序为 [w, x, y, z]。

    返回:
        np.ndarray: 旋转矩阵，形状为 (3, 3)。
    """
    w, x, y, z = q[0],q[1],q[2],q[3]


    # 计算旋转矩阵的元素
    R = np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x**2 - 2*y**2]
    ])

    return R

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


def rotationMatrixToEulerAngles(R) :    
    # sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
     
    # singular = sy < 1e-6
 
    # if  not singular :
    #     x = math.atan2(R[2,1] , R[2,2])
    #     y = math.atan2(-R[2,0], sy)
    #     z = math.atan2(R[1,0], R[0,0])
    # else :
    #     x = math.atan2(-R[1,2], R[1,1])
    #     y = math.atan2(-R[2,0], sy)
    #     z = 0

    # euler = np.array([x, y, z])
    matrix = np.eye(4)
    matrix[:3,:3] = R
    euler = euler_from_matrix(matrix)
    
    return euler


class ChessboardPoseEstimation:
    def __init__(self,fx,cx,fy,cy,target_points):
        """
        input:fx,cx,fy,cy
        """
        # 用户选择检测 ChArUco 板还是棋盘格
        tf_buffer = tf2_ros.Buffer()
        listener = tf2_ros.TransformListener(tf_buffer)
        self.detection_mode = self.choose_detection_mode()
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/rgb/image_rect_color", Image, self.image_callback,tf_buffer)
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        if self.detection_mode == "charuco":
            # ChArUco 板的参数
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
            self.charuco_board = cv2.aruco.CharucoBoard((12, 9), 0.015, 0.01125, self.aruco_dict)  # 使用新的 API
            self.charuco_params = cv2.aruco.DetectorParameters()
        else:
            # 棋盘格的参数
            self.chessboard_size = (11, 8)  # 棋盘格内部角点数量
            self.object_points = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
            self.object_points[:, :2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1, 2)*0.02

        self.camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        self.dist_coeffs = np.zeros((4, 1))  # 假设没有畸变
        self.target_points = target_points

    def choose_detection_mode(self):
        # 让用户选择检测模式
        print("请选择检测模式：")
        print("1. ChArUco 板")
        print("2. 棋盘格")
        choice = input("输入 1 或 2: ").strip()
        if choice == "1":
            return "charuco"
        elif choice == "2":
            return "chessboard"
        else:
            print("无效选择，默认使用棋盘格检测。")
            return "chessboard"

    def image_callback(self, data,tf_buffer):
        # R_base_board = np.array([[-0.98323296,0.18407529,-0.0136509 ],
        #                         [-0.18204178,-0.98275788,0.01515837],
        #                         [-0.01066477,0.01741343,0.99979192]])
        O = self.target_points[0]
        X = self.target_points[1]
        Y = self.target_points[2]

        # 计算姿态和欧拉角
        R_base_board, t, euler_angles = compute_pose_and_euler_angles(O, X, Y)
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
            return e

        if self.detection_mode == "charuco":
            R_camera_board = self.detect_charuco(cv_image)
        else:
            R_camera_board = self.detect_chessboard(cv_image)
        print("euler camera_board:",rotationMatrixToEulerAngles(R_camera_board))
        # 显示图像
        cv2.imshow("Pose Estimation", cv_image)
        cv2.waitKey(1)
        transform_base_tool0 = tf_buffer.lookup_transform('base', 'tool0_controller', rospy.Time(0), rospy.Duration(0.1))
        q_base_tool0 = [transform_base_tool0.transform.rotation.w,transform_base_tool0.transform.rotation.x,transform_base_tool0.transform.rotation.y,transform_base_tool0.transform.rotation.z]
        R_base_tool0 = quaternion_to_rotation_matrix(q_base_tool0)
         # 计算 R_base_camera
        R_base_camera = np.dot(R_base_board, R_camera_board.T)

        # 计算 R_tool0_camera
        R_tool0_camera = np.dot(R_base_tool0.T, R_base_camera)
       
        euler = rotationMatrixToEulerAngles(R_tool0_camera)
        print("R_tool0_camera",R_tool0_camera)
        print("euler",euler)



    def detect_chessboard(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)

        if ret:
            # 提高角点检测精度
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)

            # 计算位姿
            ret, rvec, tvec = cv2.solvePnP(self.object_points, corners2, self.camera_matrix, self.dist_coeffs)

            if ret:
                # 在图像上绘制坐标系
                self.draw_axis(img, rvec, tvec)
                # 在图像上显示位姿信息
                cv2.putText(img, f"Rotation: {rvec.flatten()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(img, f"Translation: {tvec.flatten()}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                R,_ =  cv2.Rodrigues(rvec)
                return R

    def detect_charuco(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.charuco_params)  # 使用新的 API
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None:
            # 检测 ChArUco 角点
            ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, self.charuco_board)

            if ret > 0:
                # 计算位姿
                ret, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, self.charuco_board, self.camera_matrix, self.dist_coeffs, None, None)
                if ret:
                    # 在图像上绘制坐标系
                    self.draw_axis(img, rvec, tvec)
                    # 在图像上显示位姿信息
                    cv2.putText(img, f"Rotation: {rvec.flatten()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(img, f"Translation: {tvec.flatten()}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    R,_ =  cv2.Rodrigues(rvec)
                return R
    
    def draw_axis(self, img, rvec, tvec, length=1):
        # 定义3D坐标系的点
        axis_points = np.float32([[0, 0, 0], [length, 0, 0], [0, length, 0], [0, 0, length]])

        # 将3D点投影到2D图像平面
        img_points, _ = cv2.projectPoints(axis_points, rvec, tvec, self.camera_matrix, self.dist_coeffs)

        # 将浮点数坐标转换为整数
        img_points = np.int32(img_points).reshape(-1, 2)

        # 绘制坐标系
        img = cv2.line(img, tuple(img_points[0]), tuple(img_points[1]), (0, 0, 255), 1)
        img = cv2.line(img, tuple(img_points[0]), tuple(img_points[2]), (0, 255, 0), 1)
        img = cv2.line(img, tuple(img_points[0]), tuple(img_points[3]), (255, 0, 0), 1)

        return img


if __name__ == '__main__':
    rospy.init_node('chessboard_pose_estimation', anonymous=True)
    # TODO load camera calibration parameters from config file
    target_points = np.load("/home/glab/Hardware_WS/src/handeye_calib/data/trial1/target_points.npy")
    chessboard_pose = ChessboardPoseEstimation(618.21,641.34,618.32,368.05,target_points)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()