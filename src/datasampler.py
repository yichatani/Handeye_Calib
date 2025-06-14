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
import json



class DataSampler:
    def __init__(self,configs):
        # 初始化ROS节点
        rospy.init_node('pose_image_saver')

        # 创建TF监听器
        self.listener = tf.TransformListener()

        # 图像桥接器
        self.bridge = CvBridge()

        # 订阅图像话题
        self.image_sub = rospy.Subscriber(configs["img_topic"], Image, self.image_callback)

        # 变量存储当前的图像和pose
        self.current_image = None
        self.current_pose = None
        self.camera_link = configs["camera_link"]
        self.base_link = configs["robot_base_link"]
        self.ee_link = configs["ee_link"]

        # 创建保存路径
        self.img_dir = configs["data_dir"] + "/images"
        self.pose_dir = configs["data_dir"] + "/poses"
        self.data_dir = configs["data_dir"]
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)
        if not os.path.exists(self.pose_dir):
            os.makedirs(self.pose_dir)

        # 获取下一个可用的序号
        self.img_counter = len(os.listdir(self.img_dir))  # 获取已有的图片数量作为起始序号
        self.pose_counter = self.img_counter  # 获取已有的姿态文件数量作为起始序号

        # 初始化DataFrame用于保存pose数据
        if self.pose_counter == 0:
            self.pose_data = pd.DataFrame(columns=["x", "y", "z", "roll", "pitch", "yaw"])
        else:
            # 读取已有的pose数据
            self.pose_data = pd.read_excel(os.path.join(self.pose_dir, f"ee2base_pose.xlsx"))
            print("self.pose_data: ", self.pose_data)

        # param for chessboard estimation
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.chessboard_size = configs["chessboard_size"]
        # 棋盘格的参数
        self.object_points = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
        self.object_points[:, :2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1, 2)*configs["chessboard_square_size"]

        self.camera_matrix = np.array([[configs["camera_matrix"]["fx"], 0, configs["camera_matrix"]["cx"]], [0, configs["camera_matrix"]["fy"], configs["camera_matrix"]["cy"]], [0, 0, 1]], dtype=np.float32)
        self.dist_coeffs = np.zeros((4, 1))  # 假设没有畸变

    def image_callback(self, msg):
        """ 回调函数，处理图像消息 """
        try:
            # 使用CvBridge将ROS图像消息转换为OpenCV图像
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.current_image = cv_image
        except Exception as e:
            rospy.logerr(f"Error converting image: {e}")

    def detect_chessboard(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)

        if ret:
            # 亚像素级角点优化
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)

            # 计算位姿
            ret, rvec, tvec = cv2.solvePnP(self.object_points, corners2, self.camera_matrix, self.dist_coeffs)

            if ret:
                # 绘制坐标系和文本
                # self.draw_axis(img, rvec, tvec)
                # cv2.putText(img, f"Rotation: {rvec.flatten()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                # cv2.putText(img, f"Translation: {tvec.flatten()}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # 转换为齐次变换矩阵
                R, _ = cv2.Rodrigues(rvec)  # 旋转向量转旋转矩阵
                T = np.eye(4)               # 初始化4x4单位矩阵
                T[:3, :3] = R               # 填充旋转部分
                T[:3, 3] = tvec.flatten()   # 填充平移部分

                return T  # 返回齐次变换矩阵
        return None

    def get_tool_pose(self, num_samples=10, threshold=0.01):
        """ 获取tool0在base_link下的姿态，并评估稳定性 """
        try:
            poses = []

            # 读取多个样本
            for _ in range(num_samples):
                (trans, rot) = self.listener.lookupTransform(self.base_link, self.ee_link, rospy.Time(0))
                # 转换旋转为欧拉角（roll, pitch, yaw）
                euler = tf.transformations.euler_from_quaternion(rot)
                roll, pitch, yaw = euler

                # 将位置从米转为毫米，旋转角度从弧度转为度
                pose = {
                    "x": trans[0],  # m
                    "y": trans[1],
                    "z": trans[2],
                    "roll": roll,  # rad
                    "pitch": pitch,
                    "yaw": yaw
                }
                poses.append(pose)
                rospy.sleep(0.1)  # 等待0.1秒，确保TF数据稳定

            # 转换为NumPy数组，便于计算
            poses_array = np.array([[p["x"], p["y"], p["z"], p["roll"], p["pitch"], p["yaw"]] for p in poses])

            # 计算每列的标准差，判断稳定性
            std_devs = np.std(poses_array, axis=0)

            rospy.loginfo(f"Standard deviations: {std_devs}")
            rospy.loginfo(f"poses_array: {poses_array}")

            # 判断标准差是否小于阈值，如果是则认为姿态稳定
            if np.all(std_devs < threshold):
                # 使用平均值作为稳定的姿态数据
                mean_pose = np.mean(poses_array, axis=0)
                stable_pose = Pose()
                stable_pose.position.x = mean_pose[0]
                stable_pose.position.y = mean_pose[1]
                stable_pose.position.z = mean_pose[2]
                stable_pose.orientation.x = mean_pose[3]
                stable_pose.orientation.y = mean_pose[4]
                stable_pose.orientation.z = mean_pose[5]

                return stable_pose
            else:
                print("Data is unstable")
                rospy.logwarn("TF data is unstable, retrying...")
                return None  # 如果数据不稳定，返回None

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn("Unable to get transform from base_link to tool0")
            return None

    def save_img_pose_sample(self):
        """ 保存当前的图像和pose """
        if self.current_image is None or self.current_pose is None:
            rospy.logwarn("No image or pose data available to save.")
            if self.current_image is None:
                print("self.current_image is None")
            else:
                print("self.current_pose is None")
            return

        # 更新计数器
        self.img_counter += 1
        self.pose_counter += 1



        # 获取稳定的姿态信息并保存到DataFrame
        # self.current_pose = self.get_tool_pose(num_samples=10, threshold=0.1)  # 使用质量评估
        if self.current_pose is not None:
            pose_data = {
                "x": self.current_pose.position.x,
                "y": self.current_pose.position.y,
                "z": self.current_pose.position.z,
                "roll": self.current_pose.orientation.x,
                "pitch": self.current_pose.orientation.y,
                "yaw": self.current_pose.orientation.z
            }

            # 将姿态数据添加到DataFrame
            # self.pose_data = self.pose_data.append(pose_data, ignore_index=True)
            self.pose_data = pd.concat([self.pose_data, pd.DataFrame([pose_data])], ignore_index=True)
            # 保存DataFrame到Excel文件
            excel_filename = os.path.join(self.pose_dir, f"ee2base_pose.xlsx")
            self.pose_data.to_excel(excel_filename, index=False)
            pose_data_nps = self.pose_data.to_numpy()
            # turn xyzrpy to T
            pose_data_matrixs = []
            for pose_data_np in pose_data_nps:
                pose_data_matrix = np.eye(4)
                pose_data_matrix[:3, :3] = tf.transformations.euler_matrix(pose_data_np[3], pose_data_np[4], pose_data_np[5])[:3, :3]
                pose_data_matrix[:3, 3] = pose_data_np[:3]
                pose_data_matrixs.append(pose_data_matrix.copy())
            pose_data_matrixs = np.array(pose_data_matrixs)
            np.save(self.pose_dir+"/ee2base_matrix.npy", pose_data_matrixs)
            # 保存图像
            img_filename = os.path.join(self.img_dir, f"image_{self.img_counter}.jpg")
            cv2.imwrite(img_filename, self.current_image)
            rospy.loginfo(f"Image saved as {img_filename}")
            rospy.loginfo(f"Pose saved as {excel_filename}")

        else:
            rospy.logwarn("Pose is unstable, not saving.")
    
    def sample_target_points(self):
        """
        take three specific points of the target calibration board and save the target orientation to a file.
        """
        point_num = 3
        point_names = ["O", "X", "Y"]
        tool_poses = []
        for i in range(point_num):
            input(f"Move end effecter to point{i+1} ({point_names[i]}) then press Enter: ").strip().lower()
            tool_pose = self.get_tool_pose(10,0.1)
            tool_position = np.array([tool_pose.position.x, tool_pose.position.y, tool_pose.position.z])
            tool_poses.append(tool_position.copy())
        tool_poses = np.array(tool_poses)
        target_points = tool_poses
        save_path = self.data_dir+"/target_points.npy"
        np.save(save_path, target_points)
        print(f"target_points saved to {save_path}")

    def sample_pose_and_image(self):
        rospy.Timer(rospy.Duration(0.1), self.check_input)
        rospy.spin()

    def check_input(self, event):
        """ 定期检查终端输入，触发保存 """
        user_input = input("Press 's' to save the current sample (image and pose) or press 'q' to quit: ").strip().lower()
        if user_input == 's':
            self.current_pose = self.get_tool_pose(num_samples=10, threshold=0.1)  # 使用质量评估
            user_input = input("Press 'y' to confirm or press 'q' to quit: ").strip().lower()
            if user_input == 'y':
                self.save_img_pose_sample()
            elif user_input == 'q':
                rospy.signal_shutdown("User requested shutdown.")
        elif user_input == 'q':
            rospy.signal_shutdown("User requested shutdown.")

    def detect_chessboard_and_save(self):
        board2camera_pose = []
        files = [f for f in os.listdir(self.img_dir) if f.startswith('image_') and f.endswith('.jpg')]
        
        # 提取数字并排序（例如：image_10.jpg -> 10）
        files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        
        # 按顺序读取文件
        for filename in files:
            filepath = os.path.join(self.img_dir, filename)
            with open(filepath, 'rb') as f:
                # 处理文件内容（例如用Pillow或OpenCV加载图片）
                print(f"Processing: {filename}")
                img = cv2.imread(os.path.join(self.img_dir, filename))
                T_board2camera =self.detect_chessboard(img)
                board2camera_pose.append(T_board2camera)
        if len(board2camera_pose) == 0:
            rospy.logwarn("No image file found.")
        board2camera_pose = np.array(board2camera_pose)
        np.save(self.data_dir+"/poses/board2camera_matrix.npy", board2camera_pose)

if __name__ == '__main__':
    config_file_path = "/home/glab/Hardware_WS/src/handeye_calib/src/config_files/trial1.json"
    with open(config_file_path, 'r', encoding='utf-8') as file:
        configs = json.load(file)
    sampler = DataSampler(configs)
    ### get target orientation
    # sampler.sample_target_points()
    # oxy = np.load("/home/glab/Hardware_WS/src/handeye_calib/data/trial1/target_points.npy")
    # print("oxy:",oxy)
    # sampler.sample_pose_and_image()
    # read images under sampler.img_dir and estimate board pose 
    # board2camera_pose = []
    # files = [f for f in os.listdir(sampler.img_dir) if f.startswith('image_') and f.endswith('.jpg')]
    
    # # 提取数字并排序（例如：image_10.jpg -> 10）
    # files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    # # 按顺序读取文件
    # for filename in files:
    #     filepath = os.path.join(sampler.img_dir, filename)
    #     with open(filepath, 'rb') as f:
    #         # 处理文件内容（例如用Pillow或OpenCV加载图片）
    #         print(f"Processing: {filename}")
    #         img = cv2.imread(os.path.join(sampler.img_dir, filename))
    #         T_board2camera =sampler.detect_chessboard(img)
    #         board2camera_pose.append(T_board2camera)
    
    # board2camera_pose = np.array(board2camera_pose)
    # np.save(sampler.data_dir+"/poses/board2camera_matrix.npy", board2camera_pose)
    sampler.detect_chessboard_and_save()

    
        
        
        

   
    