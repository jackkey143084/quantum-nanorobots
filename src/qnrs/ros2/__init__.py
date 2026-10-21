"""ROS 2 integration for QNRS.

``bridge_logic`` is pure Python (testable without ROS); the ament package under
``ros2_ws/src/qnrs_ros2`` wraps it in an ``rclpy`` node with messages/services.
"""

from qnrs.ros2.bridge_logic import BridgeStatus, QuantumBridge

__all__ = ["BridgeStatus", "QuantumBridge"]
