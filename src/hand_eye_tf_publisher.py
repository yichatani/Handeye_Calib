#!/home/glab/miniconda3/envs/py3-mink-env/bin/python3
import rospy
from dynamic_reconfigure.server import Server
import tf
from geometry_msgs.msg import TransformStamped
import sys
import os
# TODO change the path here
from handeye_calib.cfg import handeye_paramConfig 
fix_config = True
# 创建一个全局变量，用于存储动态配置的参数
# 这里的参数是默认参数，在不使用动态调参时发布这个参数

import sys
print("all paths:",sys.path)
# TODO read from file
current_config = {
    "roll": 0.0,
    "pitch": 0.00359,
    "yaw": 0.0039,
    "x": 0.03705,
    "y": -0.09919999999999998,
    "z": 0.09119999999999999
}

# fixed_config = {
#     "roll": 0.15 - 0.15,
#     "pitch": 0.008-0.0,
#     "yaw": -0.01026 + 0.0,
#     "x": 0.03081 + 0.0,
#     "y": -0.0903,
#     "z": 0.18599 - 0.07
# }

# fixed_config = {
#     "roll": 0.15 - 0.15,
#     "pitch":-0.0095-0.0,
#     "yaw": -0.0042 + 0.0,
#     "x": 0.03588 + 0.0,
#     "y": -0.08959,
#     "z": 0.18999 - 0.07
# }

fixed_config = {
    "roll": 0.15 - 0.15,
    "pitch":-0.0095-0.0,
    "yaw": -0.0042 + 0.0,
    "x": 0.033788 + 0.0,
    "y": -0.08449,
    "z": 0.1134
}

# 动态配置回调函数
def dynamic_reconfigure_callback(config, level):
    global current_config
    
    # 更新动态配置的参数
    current_config["roll"] = config["roll"] - 0.15
    current_config["pitch"] = config["pitch"] 
    current_config["yaw"] = config["yaw"] 
    current_config["x"] = config["x"] + 0.0
    current_config["y"] = config["y"]
    current_config["z"] = config["z"] - 0.07             
    return config

# 发布 tf 变换的函数
def publish_tf(event):
    # 获取当前的配置
    if not fix_config:
        x = current_config["x"]
        y = current_config["y"]
        z = current_config["z"]
        roll = current_config["roll"]
        pitch = current_config["pitch"]
        yaw = current_config["yaw"]
    else:
        x = fixed_config["x"]
        y = fixed_config["y"]
        z = fixed_config["z"]
        roll = fixed_config["roll"] 
        pitch = fixed_config["pitch"]
        yaw = fixed_config["yaw"]
    # 创建 tf 变换广播器
    br = tf.TransformBroadcaster()
    
    # 构造 tf 变换消息
    t = TransformStamped()
    t.header.stamp = rospy.Time.now()
    t.header.frame_id = "tool0_controller"  # 可以根据需要调整
    t.child_frame_id = "rgb_camera_link"  # 机器人末端的frame
    
    # 使用当前的配置设置平移
    t.transform.translation.x = x
    t.transform.translation.y = y
    t.transform.translation.z = z  # 假设在z轴也有位移
    
    # 使用当前的配置设置旋转（欧拉角转四元数）
    q = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
    t.transform.rotation.x = q[0]
    t.transform.rotation.y = q[1]
    t.transform.rotation.z = q[2]
    t.transform.rotation.w = q[3]
    
    # 正确的发布变换方式
    br.sendTransform(
        (t.transform.translation.x, t.transform.translation.y, t.transform.translation.z),  # translation
        (t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w),  # rotation (quaternion)
        t.header.stamp,  # 时间戳
        t.child_frame_id,  # child frame
        t.header.frame_id  # parent frame
    )
    
    # rospy.loginfo("Published tf: translation ({}, {}, {}), rotation (quaternion): ({}, {}, {}, {})".format(
    #     t.transform.translation.x, t.transform.translation.y, t.transform.translation.z,
    #     t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w))

if __name__ == "__main__":
    # 初始化 ROS 节点
    rospy.init_node("aruco_tracker_server")

    # 创建动态配置服务
    server = Server(handeye_paramConfig, dynamic_reconfigure_callback)

    # 设置定时器，每0.1秒发布一次 tf 变换
    rospy.Timer(rospy.Duration(0.05), publish_tf)

    # 保持节点运行
    rospy.spin()