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

class PoseImageSaver:
    def __init__(self,img_dir,pose_dir):
        # 初始化ROS节点
        rospy.init_node('pose_image_saver')

        # 创建TF监听器
        self.listener = tf.TransformListener()

        # 图像桥接器
        self.bridge = CvBridge()

        # 订阅图像话题
        self.image_sub = rospy.Subscriber('/rgb/image_raw', Image, self.image_callback)

        # 变量存储当前的图像和pose
        self.current_image = None
        self.current_pose = None

        # 创建保存路径
        self.img_dir = img_dir
        self.pose_dir = pose_dir
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
            self.pose_data = pd.read_excel(os.path.join(self.pose_dir, f"pose_data.xlsx"))
            print("self.pose_data: ", self.pose_data)

        # 按钮触发采样（使用终端输入）
        rospy.Timer(rospy.Duration(0.1), self.check_input)

    def image_callback(self, msg):
        """ 回调函数，处理图像消息 """
        try:
            # 使用CvBridge将ROS图像消息转换为OpenCV图像
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.current_image = cv_image
        except Exception as e:
            rospy.logerr(f"Error converting image: {e}")

    def get_tool_pose(self, num_samples=10, threshold=0.01):
        """ 获取tool0在base_link下的姿态，并评估稳定性 """
        try:
            poses = []
            # 读取多个样本
            for _ in range(num_samples):
                (trans, rot) = self.listener.lookupTransform('base_link', 'tool0', rospy.Time(0))
                # 转换旋转为欧拉角（roll, pitch, yaw）
                euler = tf.transformations.euler_from_quaternion(rot)
                roll, pitch, yaw = euler

                # 将位置从米转为毫米，旋转角度从弧度转为度
                pose = {
                    "x": trans[0] * 1000,  # 转为mm
                    "y": trans[1] * 1000,
                    "z": trans[2] * 1000,
                    "roll": roll * 180 / np.pi,  # 转为度
                    "pitch": pitch * 180 / np.pi,
                    "yaw": yaw * 180 / np.pi
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

    def save_sample(self):
        """ 保存当前的图像和pose """
        if self.current_image is None or self.current_pose is None:
            rospy.logwarn("No image or pose data available to save.")
            return

        # 更新计数器
        self.img_counter += 1
        self.pose_counter += 1

        # 保存图像
        img_filename = os.path.join(self.img_dir, f"image_{self.img_counter}.jpg")
        cv2.imwrite(img_filename, self.current_image)
        rospy.loginfo(f"Image saved as {img_filename}")

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
            self.pose_data = self.pose_data.append(pose_data, ignore_index=True)

            # 保存DataFrame到Excel文件
            excel_filename = os.path.join(self.pose_dir, f"pose_data.xlsx")
            self.pose_data.to_excel(excel_filename, index=False)
            rospy.loginfo(f"Pose saved as {excel_filename}")
        else:
            rospy.logwarn("Pose is unstable, not saving.")

    def check_input(self, event):
        """ 定期检查终端输入，触发保存 """
        user_input = input("Press 's' to save the current sample (image and pose): ").strip().lower()
        if user_input == 's':
            self.current_pose = self.get_tool_pose(num_samples=10, threshold=0.1)  # 使用质量评估
            user_input = input("Press 'y' to confirm: ").strip().lower()
            if user_input == 'y':
                self.save_sample()

    







if __name__ == "__main__":
    try:
        saver = PoseImageSaver("imgs","poses")
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
