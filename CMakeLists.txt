cmake_minimum_required(VERSION 3.0.2)
project(handeye_calib)

find_package(catkin REQUIRED COMPONENTS
  dynamic_reconfigure
)

# 启用Python模块支持
catkin_python_setup()

# 生成动态参数配置
generate_dynamic_reconfigure_options(
  cfg/handeye_param.cfg
)

# 声明动态参数生成的依赖
add_dependencies(${PROJECT_NAME}_gencfg ${${PROJECT_NAME}_EXPORTED_TARGETS})

catkin_package()

# # 安装Python脚本
catkin_install_python(PROGRAMS
  src/hand_eye_tf_publisher.py
  # scripts/estimate_chessboard.py
  src/
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# install(
#   FILES scripts/__init__.py
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/scripts
# )

