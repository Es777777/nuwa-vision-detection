# nuwa_vision 本地 ROS 2 包

这个包按 `rc2026_ros2_ws` 常见的 `src/package/launch/config` 形式组织，远端仓库只作为目录结构参考，没有在远端目录写入、提交或推送。

包目录：`/home/koji/桌面/识别方案/ros2_ws/src/nuwa_vision`。本地工作区已经用 `colcon build --base-paths ros2_ws --build-base ros2_ws/build --install-base ros2_ws/install` 编译通过。

## 输入和输出

- 输入：`camera/color/image_raw`（`sensor_msgs/Image`）、`camera/color/camera_info`
- 输出：`vision/ball_pose`（像素坐标暂存于 `PoseStamped.position.x/y`，置信度在 `z`）、`vision/ball_column_state`
- 启动：`ros2 launch nuwa_vision nuwa_vision.launch.py`

启动前执行：`source /home/koji/桌面/识别方案/ros2_ws/install/setup.bash`。

当前节点是可运行的 RGB 候选层。真实球心三维坐标、柱口平面和 D435i 深度图接入后，调用 `classify_ball_column`：同时检查球心到柱轴距离和球底到柱口平面的间隙，输出 `ON_COLUMN`、`OFF_COLUMN` 或 `UNKNOWN`。没有深度时不把“没看到柱”误判成离柱。

`detector.py` 已包含实际深度几何：`depth_points` 用内参反投影，`fit_sphere` 固定球半径 0.100 m 拟合球心；柱模型使用外半径 0.135 m、杯口内半径 0.090 m、凹槽深度 0.100 m。`classify_ball_column` 的判据是球心到柱轴小于 35 mm、球底到杯口平面间隙小于 45 mm，并要求深度拟合有效。节点订阅 `camera/aligned_depth_to_color/image_raw`，支持 D435i 常见的 `32FC1` 米制和 `16UC1` 毫米制深度。

最终部署策略固定为：RGB 颜色/霍夫或椭圆只产生候选；深度球面拟合确认球；深度平面和圆柱拟合确认柱及杯口；用上述三维关系区分 `ON_COLUMN`、`OFF_COLUMN`、`UNKNOWN`；连续 3 帧确认、连续 5 帧无效进入丢失状态。颜色和边缘不直接发布位置。

## 评测边界

现有 Blender/Gazebo 图集是 RGB 渲染，不能证明 D435i 深度精度。当前离线基线仍是：正样本 60 张，多参数霍夫圆 53/60（88.3%）；鲁棒集 192 张 135/192（70.3%）；融合 V2 为 131/192（68.2%）。因此报告中应保留“融合尚未超过霍夫圆”的结论，待接入真实 RGB-D 或带深度的仿真后再更新数字。
