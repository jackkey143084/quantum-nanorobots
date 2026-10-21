"""rclpy node exposing the QNRS control plane on the ROS 2 graph.

Topics
------
``/qnrs/telemetry``     std_msgs/String         runtime telemetry (JSON)
``/qnrs/state``         qnrs_ros2/QuantumState  per-device state (JSON-driven)
``/qnrs/capabilities``  qnrs_ros2/DeviceCapabilities

Services
--------
``/qnrs/negotiate``      qnrs_ros2/NegotiateProtocol
``/qnrs/execute_pulse``  qnrs_ros2/ExecutePulseSequence
``/qnrs/submit_circuit`` qnrs_ros2/SubmitCircuit

The node owns a background asyncio loop because the QNRS runtime is async; the
ROS callbacks schedule coroutines onto that loop and wait for the result.
"""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any, Coroutine, Optional

from qnrs.ros2.bridge_logic import QuantumBridge

try:  # pragma: no cover - requires a ROS 2 installation
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String

    from qnrs_ros2.msg import DeviceCapabilities, QuantumState
    from qnrs_ros2.srv import ExecutePulseSequence, NegotiateProtocol, SubmitCircuit

    ROS_AVAILABLE = True
except Exception:  # pragma: no cover
    ROS_AVAILABLE = False
    Node = object  # type: ignore[assignment,misc]


class QuantumBridgeNode(Node):  # type: ignore[misc]
    """Thin ROS shell around :class:`qnrs.ros2.bridge_logic.QuantumBridge`."""

    def __init__(self, bridge: Optional[QuantumBridge] = None) -> None:
        if not ROS_AVAILABLE:
            raise RuntimeError("rclpy is not available; source your ROS 2 environment first")
        super().__init__("quantum_bridge")
        self.bridge = bridge or QuantumBridge()
        self.declare_parameter("default_device", "numpy:simulator")
        self.declare_parameter("publish_period_s", 5.0)

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

        self._telemetry_pub = self.create_publisher(String, "/qnrs/telemetry", 10)
        self._state_pub = self.create_publisher(QuantumState, "/qnrs/state", 10)
        self._caps_pub = self.create_publisher(DeviceCapabilities, "/qnrs/capabilities", 10)

        self.create_service(NegotiateProtocol, "/qnrs/negotiate", self._handle_negotiate)
        self.create_service(ExecutePulseSequence, "/qnrs/execute_pulse", self._handle_execute_pulse)
        self.create_service(SubmitCircuit, "/qnrs/submit_circuit", self._handle_submit_circuit)

        period = float(self.get_parameter("publish_period_s").value)
        self.create_timer(period, self._publish_telemetry)
        self.get_logger().info("QNRS ROS 2 bridge started")

    # ------------------------------------------------------------------ plumbing
    def _run_async(self, coro: Coroutine[Any, Any, Any]) -> Any:
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout=30.0)

    def _device(self) -> str:
        return str(self.get_parameter("default_device").value)

    # ------------------------------------------------------------------ publishers
    def _publish_telemetry(self) -> None:
        message = String()
        message.data = self.bridge.telemetry_json()
        self._telemetry_pub.publish(message)
        try:
            state = self._run_async(self.bridge.state_message(self._device()))
        except Exception as exc:
            self.get_logger().warn(f"state publish failed: {exc}")
            return
        state_msg = QuantumState()
        state_msg.stamp = float(state["stamp"])
        state_msg.device_id = state["device_id"]
        state_msg.num_qubits = int(state["num_qubits"])
        state_msg.fidelity = float(state["fidelity"])
        state_msg.predicted_fidelity = float(state["predicted_fidelity"])
        state_msg.coherence_level = str(state["coherence_level"])
        state_msg.age_us = float(state["age_us"])
        state_msg.t1_us = float(state["t1_us"])
        state_msg.t2_us = float(state["t2_us"])
        state_msg.requires_correction = state["coherence_level"] in ("degraded", "lost")
        self._state_pub.publish(state_msg)

    # ------------------------------------------------------------------ services
    def _handle_negotiate(self, request: Any, response: Any) -> Any:
        result = self._run_async(
            self.bridge.negotiate(request.device_id, list(request.required_protocols), request.allow_fallback)
        )
        response.success = bool(result["success"])
        response.protocol = str(result["protocol"])
        response.score = float(result["score"])
        response.message = str(result["message"])
        return response

    def _handle_execute_pulse(self, request: Any, response: Any) -> Any:
        sequence = json.loads(request.sequence_json or "{}")
        result = self._run_async(
            self.bridge.execute_pulse(sequence, request.device_id or None)
        )
        response.success = bool(result["success"])
        response.job_id = str(result["job_id"])
        response.device_id = str(result["device_id"])
        response.counts_json = str(result["counts_json"])
        response.excitation = float(result["excitation"])
        response.message = str(result["message"])
        return response

    def _handle_submit_circuit(self, request: Any, response: Any) -> Any:
        program = json.loads(request.program_json or "{}")
        result = self._run_async(
            self.bridge.submit_circuit(program, request.device_id or None)
        )
        response.success = bool(result["success"])
        response.job_id = str(result["job_id"])
        response.device_id = str(result["device_id"])
        response.counts_json = str(result["counts_json"])
        response.fidelity = float(result["fidelity"])
        response.message = str(result["message"])
        return response

    # ------------------------------------------------------------------ shutdown
    def destroy_node(self) -> bool:  # pragma: no cover
        try:
            asyncio.run_coroutine_threadsafe(self.bridge.close(), self._loop).result(timeout=10.0)
        except Exception:
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5.0)
        return super().destroy_node()


def main(args: Optional[list] = None) -> None:  # pragma: no cover
    if not ROS_AVAILABLE:
        raise SystemExit("rclpy not found; source /opt/ros/<distro>/setup.bash first")
    rclpy.init(args=args)
    node = QuantumBridgeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":  # pragma: no cover
    main()
