from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
def generate_launch_description():
    cfg = os.path.join(get_package_share_directory('nuwa_vision'), 'config', 'nuwa_vision.yaml')
    return LaunchDescription([Node(package='nuwa_vision', executable='rgbd_detector', name='nuwa_rgbd_detector', parameters=[cfg])])
