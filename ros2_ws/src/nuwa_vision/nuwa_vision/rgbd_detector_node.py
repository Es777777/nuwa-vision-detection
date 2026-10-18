import json

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, PointStamped
from std_msgs.msg import String
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
import tf2_ros

from .detector import CameraModel, analyze_rgbd
from .state_machine import TrackFilter, FieldMotionGate


class RGBDDetector(Node):
    """RGB-D detector using the same proposal and gates as offline evaluation."""

    def __init__(self):
        super().__init__('nuwa_rgbd_detector')
        self.bridge = CvBridge()
        self.last_info = None
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.track_filter = TrackFilter(acquire=3, drop=5)
        self.declare_parameter('max_speed_mps', 2.5)
        self.motion_gate = FieldMotionGate(
            max_speed_mps=float(self.get_parameter('max_speed_mps').value))

        self.ball_pub = self.create_publisher(PoseStamped, 'vision/ball_pose', 10)
        self.column_pub = self.create_publisher(PointStamped, 'vision/column_xy_center', 10)
        self.state_pub = self.create_publisher(String, 'vision/ball_column_state', 10)
        self.ball_mask_pub = self.create_publisher(Image, 'vision/ball_mask', 10)
        self.column_mask_pub = self.create_publisher(Image, 'vision/column_mask', 10)
        self.depth_roi_pub = self.create_publisher(Image, 'vision/depth_roi_mask', 10)
        self.debug_pub = self.create_publisher(String, 'vision/detection_debug', 10)

        self.rgb_sub = Subscriber(self, Image, 'camera/color/image_raw')
        self.depth_sub = Subscriber(self, Image, 'camera/aligned_depth_to_color/image_raw')
        self.sync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], queue_size=8, slop=0.03)
        self.sync.registerCallback(self.rgbd_cb)
        self.create_subscription(
            CameraInfo, 'camera/color/camera_info', self.camera_info_cb, 10)
        self.declare_parameter('roi', [0, 0, 0, 0])
        self.declare_parameter('target_frame', 'field')

    def camera_info_cb(self, msg):
        self.last_info = msg

    @staticmethod
    def transform_matrix(tf):
        tr = tf.transform.translation
        q = tf.transform.rotation
        x, y, z, w = float(q.x), float(q.y), float(q.z), float(q.w)
        rot = np.array([
            [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
        ], dtype=np.float64)
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = rot
        matrix[:3, 3] = [float(tr.x), float(tr.y), float(tr.z)]
        return matrix

    def rgbd_cb(self, msg, depth_msg):
        def stamp(value):
            return float(value.sec) + float(value.nanosec) * 1e-9

        dt = abs(stamp(msg.header.stamp) - stamp(depth_msg.header.stamp))
        if dt > 0.030:
            self.get_logger().warning(
                f'RGB/depth timestamp delta {dt:.4f}s exceeds 30 ms; frame dropped')
            return
        if (msg.header.frame_id and depth_msg.header.frame_id and
                msg.header.frame_id != depth_msg.header.frame_id):
            self.get_logger().warning(
                f'RGB/depth frame_id mismatch: {msg.header.frame_id} vs '
                f'{depth_msg.header.frame_id}; frame dropped')
            return
        try:
            image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            if depth_msg.encoding == '32FC1':
                depth = self.bridge.imgmsg_to_cv2(depth_msg, '32FC1').astype(np.float32)
            elif depth_msg.encoding == '16UC1':
                depth = self.bridge.imgmsg_to_cv2(depth_msg, '16UC1').astype(np.float32) / 1000.0
            else:
                raise ValueError(f'unsupported depth encoding {depth_msg.encoding}')
        except Exception as exc:
            self.get_logger().warning(f'RGB-D conversion failed: {exc}')
            return
        if depth.shape[:2] != image.shape[:2]:
            self.get_logger().warning(
                f'aligned depth size {depth.shape[1]}x{depth.shape[0]} != RGB '
                f'{image.shape[1]}x{image.shape[0]}; frame dropped')
            return
        if self.last_info is None or (self.last_info.width, self.last_info.height) != (
                image.shape[1], image.shape[0]):
            self.get_logger().warning(
                'CameraInfo missing or resolution does not match RGB; frame dropped')
            return

        source_frame = msg.header.frame_id or depth_msg.header.frame_id
        target_frame = str(self.get_parameter('target_frame').value)
        camera_to_field = None
        transform_error = None
        if source_frame:
            try:
                tf = self.tf_buffer.lookup_transform(
                    target_frame, source_frame, Time.from_msg(msg.header.stamp))
                camera_to_field = self.transform_matrix(tf)
            except Exception as exc:
                transform_error = str(exc)

        roi = list(self.get_parameter('roi').value)
        roi = None if roi == [0, 0, 0, 0] else tuple(roi)
        cam = CameraModel(self.last_info.k[0], self.last_info.k[4],
                          self.last_info.k[2], self.last_info.k[5])
        ball, col, ball_mask, column_mask, depth_roi, geometry = analyze_rgbd(
            image, depth, cam, camera_to_field, roi)

        temporal_ok = True
        temporal_reason = 'not_applied'
        if camera_to_field is not None and geometry['ball_center_3d'] is not None:
            frame_time = stamp(msg.header.stamp)
            temporal_ok, temporal_reason = self.motion_gate.update(
                geometry['ball_center_3d'], frame_time)
            if not temporal_ok:
                geometry['ball_center_3d'] = None
                geometry['state'] = 'UNKNOWN'
                geometry['reason'] = 'temporal_motion_gate'

        for publisher, mask in ((self.ball_mask_pub, ball_mask),
                                (self.column_mask_pub, column_mask),
                                (self.depth_roi_pub, depth_roi)):
            out = self.bridge.cv2_to_imgmsg(mask, 'mono8')
            out.header = msg.header
            publisher.publish(out)

        measured = geometry['ball_center_3d'] is not None
        track_state = self.track_filter.update(measured).value
        state_msg = String()
        state_msg.data = geometry['state']
        self.state_pub.publish(state_msg)

        frame_id = target_frame if camera_to_field is not None else source_frame
        point = geometry['ball_center_3d']
        if point is not None and frame_id:
            pose = PoseStamped()
            pose.header = msg.header
            pose.header.frame_id = frame_id
            pose.pose.orientation.w = 1.0
            pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = [
                float(v) for v in point]
            self.ball_pub.publish(pose)

        column_point = geometry['column_xy_center_3d']
        if column_point is not None and camera_to_field is not None:
            column_msg = PointStamped()
            column_msg.header = msg.header
            column_msg.header.frame_id = target_frame
            column_msg.point.x, column_msg.point.y, column_msg.point.z = [
                float(v) for v in column_point]
            self.column_pub.publish(column_msg)

        debug = {k: v for k, v in geometry.items()
                 if k not in ('ball_mask', 'column_mask', 'depth_roi_mask')}
        debug.update({'track_state': track_state, 'rgb_depth_delta_s': dt,
                      'coordinate_frame': frame_id,
                      'temporal_gate_ok': temporal_ok,
                      'temporal_gate_reason': temporal_reason,
                      'camera_transform_available': camera_to_field is not None,
                      'transform_error': transform_error,
                      'ball_pixel': None if ball is None else [float(ball.u), float(ball.v)],
                      'column_pixel_candidate': None if col is None else [float(col.u), float(col.v)]})
        self.debug_pub.publish(String(data=json.dumps(debug, ensure_ascii=False)))


def main(args=None):
    rclpy.init(args=args)
    node = RGBDDetector()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
