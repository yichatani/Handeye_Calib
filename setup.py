# from distutils.core import setup
# from catkin_pkg.python_setup import generate_distutils_setup

# # 自动从package.xml中提取元数据，并指定Python包路径
# setup_args = generate_distutils_setup(
#     packages=['handeye_calib'],  # 替换为您的Python包名
#     package_dir={'': 'scripts'}          # 指定Python包位于src目录下
# )
# setup(**setup_args)

from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(
    packages=['calib_tools'],
    package_dir={'': 'scripts'},  # 若Python模块在src目录下
    # scripts=['scripts/estimate_chessboard.py'],
    install_requires=['rospy', 'numpy'],  # 添加第三方依赖
)
setup(**setup_args)