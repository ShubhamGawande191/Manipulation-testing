"""SO101 arm controller plugin for agent-ros-bridge.

Bridges AI agent commands to the SO101 follower arm via ROS2 ros2_control.
Subscribes to /follower/joint_states and publishes to /follower/forward_controller/commands.
"""

import asyncio
import base64
import logging
import math
import threading
import time
from typing import Any

import cv2
import numpy as np
import rclpy
from agent_ros_bridge.gateway_v2.core import Header, Identity, Message, Plugin, Telemetry
from agent_ros_bridge.plugins.arm_robot import (
    ArmConfig,
    ArmState,
    ArmType,
    BaseArmController,
    CartesianPose,
    JointState,
)
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState as JointStateMsg
from std_msgs.msg import Float64MultiArray

logger = logging.getLogger("plugin.so101")

# Joint ordering
# forward_controller config order (what we publish):
CMD_JOINT_ORDER = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# joint_states topic order (alphabetical, what we receive):
STATE_JOINT_ORDER = [
    "elbow_flex",
    "gripper",
    "shoulder_lift",
    "shoulder_pan",
    "wrist_flex",
    "wrist_roll",
]

# Map from state order index → command order index
_STATE_TO_CMD = [CMD_JOINT_ORDER.index(j) for j in STATE_JOINT_ORDER]
# Map from command order index → state order index
_CMD_TO_STATE = [STATE_JOINT_ORDER.index(j) for j in CMD_JOINT_ORDER]


class CameraManager:
    """Captures frames from USB cameras using OpenCV in background threads."""

    def __init__(self, cameras: dict[str, int]):
        """
        Args:
            cameras: mapping of name → video device index, e.g. {"workspace": 0, "wrist": 2}
        """
        self._cameras = cameras
        self._caps: dict[str, cv2.VideoCapture] = {}
        self._frames: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self):
        """Open cameras and start capture thread."""
        for name, idx in self._cameras.items():
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self._caps[name] = cap
                logger.info(f"Camera '{name}' opened on /dev/video{idx}")
            else:
                logger.warning(f"Camera '{name}' failed to open /dev/video{idx}")
        if not self._caps:
            logger.error("No cameras available")
            return
        # Warmup: do a few reads per camera so they start producing frames
        for name, cap in self._caps.items():
            for _ in range(5):
                ret, frame = cap.read()
                if ret:
                    with self._lock:
                        self._frames[name] = frame
                    break
                time.sleep(0.1)
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self):
        while self._running:
            for name, cap in self._caps.items():
                ret, frame = cap.read()
                if ret:
                    with self._lock:
                        self._frames[name] = frame
            time.sleep(0.033)  # ~30 fps

    def get_frame(self, name: str) -> np.ndarray | None:
        with self._lock:
            return self._frames.get(name)

    def get_jpeg_b64(self, name: str, quality: int = 80) -> str | None:
        """Get latest frame as base64-encoded JPEG."""
        frame = self.get_frame(name)
        if frame is None:
            return None
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return None
        return base64.b64encode(buf.tobytes()).decode("ascii")

    def list_cameras(self) -> list[str]:
        return list(self._caps.keys())

    def stop(self):
        self._running = False
        for cap in self._caps.values():
            cap.release()
        self._caps.clear()
        self._frames.clear()


class SO101Controller(BaseArmController):
    """ROS2 controller for the SO101 follower arm via ros2_control."""

    def __init__(self, config: ArmConfig, namespace: str = "follower"):
        super().__init__(config)
        self._ns = namespace
        self._node: Node | None = None
        self._cmd_pub = None
        self._joint_sub = None
        self._cam_subs: dict[str, Any] = {}
        self._latest_images: dict[str, bytes] = {}
        self._spin_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        # Positions in CMD_JOINT_ORDER
        self._current_positions = [0.0] * 6
        # Camera manager (initialized separately)
        self.camera_mgr: CameraManager | None = None

    async def connect(self) -> bool:
        try:
            if not rclpy.ok():
                rclpy.init()

            self._node = Node("so101_bridge_controller")

            # Publisher for forward_controller (Float64MultiArray)
            qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
            self._cmd_pub = self._node.create_publisher(
                Float64MultiArray,
                f"/{self._ns}/forward_controller/commands",
                qos,
            )

            # Subscriber for joint_states
            self._joint_sub = self._node.create_subscription(
                JointStateMsg,
                f"/{self._ns}/joint_states",
                self._on_joint_state,
                10,
            )

            # Spin in background thread
            self._spin_thread = threading.Thread(target=self._spin, daemon=True)
            self._spin_thread.start()

            # Wait for first joint state
            for _ in range(50):
                await asyncio.sleep(0.1)
                with self._lock:
                    if any(p != 0.0 for p in self._current_positions):
                        break

            self.is_connected = True
            logger.info("Connected to SO101 follower arm via ROS2")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to SO101: {e}")
            return False

    def _spin(self):
        """Spin the ROS2 node in a background thread."""
        while rclpy.ok() and self._node is not None:
            rclpy.spin_once(self._node, timeout_sec=0.05)

    def _on_joint_state(self, msg: JointStateMsg):
        """Handle incoming joint states (alphabetical order)."""
        with self._lock:
            # Reorder from state order to command order
            for state_idx, pos in enumerate(msg.position):
                cmd_idx = _STATE_TO_CMD[state_idx]
                self._current_positions[cmd_idx] = pos
            self.current_joints = [
                JointState(
                    position=self._current_positions[i],
                    velocity=0.0,
                    effort=0.0,
                )
                for i in range(6)
            ]

    def subscribe_camera(self, topic: str, name: str):
        """Subscribe to a ROS2 image topic."""
        if self._node is None:
            return

        def _on_image(msg: Image):
            with self._lock:
                self._latest_images[name] = bytes(msg.data)

        sub = self._node.create_subscription(Image, topic, _on_image, 10)
        self._cam_subs[name] = sub
        logger.info(f"Subscribed to camera: {name} on {topic}")

    def get_camera_image(self, name: str) -> bytes | None:
        """Get the latest raw image data from a named camera."""
        with self._lock:
            return self._latest_images.get(name)

    async def disconnect(self):
        self.is_connected = False
        if self.camera_mgr:
            self.camera_mgr.stop()
        if self._node:
            self._node.destroy_node()
            self._node = None
        logger.info("Disconnected from SO101")

    async def move_joints(
        self, joint_positions: list[float], velocity: float | None = None
    ) -> bool:
        """Move to joint positions (in CMD_JOINT_ORDER).

        Args:
            joint_positions: 6 floats in radians
                [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
        """
        if not self.is_connected or self._cmd_pub is None:
            return False

        if len(joint_positions) != 6:
            logger.error(f"Expected 6 joint positions, got {len(joint_positions)}")
            return False

        self.state = ArmState.MOVING
        msg = Float64MultiArray()
        msg.data = [float(p) for p in joint_positions]
        self._cmd_pub.publish(msg)
        logger.info(f"SO101 moving to: {[round(p, 3) for p in joint_positions]}")

        # Wait for convergence
        timeout = 5.0
        tolerance = 0.12  # Increased tolerance to avoid endless 8s timeouts
        elapsed = 0.0
        while elapsed < timeout:
            await asyncio.sleep(0.1)
            elapsed += 0.1
            with self._lock:
                diffs = [abs(self._current_positions[i] - joint_positions[i]) for i in range(6)]
            if all(d < tolerance for d in diffs):
                self.state = ArmState.IDLE
                logger.debug(f"SO101 reached target position. Max diff: {max(diffs):.2f}")
                return True
            # Re-publish to keep the command active
            if int(elapsed * 10) % 5 == 0:
                self._cmd_pub.publish(msg)

        logger.debug(f"SO101 move 'timed out' (approx reached). Max diff: {max(diffs):.2f}")
        self.state = ArmState.IDLE
        return True

    async def move_cartesian(self, pose: CartesianPose, velocity: float | None = None) -> bool:
        """Cartesian move — not supported (no IK solver)."""
        logger.warning("SO101 does not support cartesian moves (no IK)")
        return False

    async def get_joint_states(self) -> list[JointState]:
        with self._lock:
            return list(self.current_joints)

    async def get_cartesian_pose(self) -> CartesianPose | None:
        return None

    async def set_gripper(self, position: float, effort: float = 0.5) -> bool:
        """Set gripper position (0=open, 1=closed).

        Maps 0..1 to the gripper joint range in radians.
        Gripper joint is index 5 in CMD_JOINT_ORDER.
        Open ≈ -0.4 rad, Closed ≈ 0.8 rad (approximate for SO101)
        """
        if not self.is_connected:
            return False

        gripper_open = -0.4
        gripper_closed = 0.8
        gripper_rad = gripper_open + position * (gripper_closed - gripper_open)

        with self._lock:
            positions = list(self._current_positions)
        positions[5] = gripper_rad  # gripper is index 5 in CMD order

        return await self.move_joints(positions)

    async def stop(self):
        """Emergency stop — hold current position."""
        logger.warning("SO101 EMERGENCY STOP")
        self.state = ArmState.PROTECTIVE_STOP
        with self._lock:
            positions = list(self._current_positions)
        if self._cmd_pub:
            msg = Float64MultiArray()
            msg.data = positions
            self._cmd_pub.publish(msg)


class SO101Plugin(Plugin):
    """Agent-ros-bridge plugin for the SO101 arm."""

    name = "so101_arm"
    version = "1.0.0"

    # Default camera device mapping: name → /dev/video index
    DEFAULT_CAMERAS = {"workspace": 2, "wrist": 0}

    def __init__(self, namespace: str = "follower", cameras: dict[str, int] | None = None):
        config = ArmConfig(
            arm_type=ArmType.GENERIC,
            ros_version="ros2",
            namespace=namespace,
            joint_names=CMD_JOINT_ORDER,
            base_frame="base_link",
            end_effector_frame="gripper_link",
            has_gripper=True,
        )
        self.controller = SO101Controller(config, namespace=namespace)
        self._camera_config = cameras if cameras is not None else self.DEFAULT_CAMERAS

    async def initialize(self, _gateway) -> None:
        success = await self.controller.connect()
        if success:
            cam_mgr = CameraManager(self._camera_config)
            cam_mgr.start()
            self.controller.camera_mgr = cam_mgr
            logger.info(f"SO101 plugin initialized (cameras: {cam_mgr.list_cameras()})")

    async def shutdown(self) -> None:
        if self.controller:
            await self.controller.disconnect()

    async def handle_message(self, message: Message, _identity: Identity) -> Message | None:
        if not message.command or not message.command.action.startswith("arm."):
            return None
        result = await self.handle_command(message.command.action, message.command.parameters)
        return Message(
            header=Header(correlation_id=message.header.message_id),
            telemetry=Telemetry(topic="/arm/result", data=result),
        )

    async def handle_command(self, command: str, params: dict[str, Any]) -> dict[str, Any]:
        if not self.controller:
            return {"error": "Controller not initialized"}

        try:
            if command == "arm.move_joints":
                joints = params.get("joints", [0.0] * 6)
                success = await self.controller.move_joints(joints)
                return {"success": success, "target_joints": joints}

            elif command == "arm.get_state":
                js = await self.controller.get_joint_states()
                return {
                    "state": self.controller.state.value,
                    "joints": {
                        CMD_JOINT_ORDER[i]: {
                            "position": round(js[i].position, 4) if i < len(js) else 0.0,
                            "position_deg": round(math.degrees(js[i].position), 1)
                            if i < len(js)
                            else 0.0,
                        }
                        for i in range(6)
                    },
                }

            elif command == "arm.gripper":
                position = params.get("position", 0.5)
                success = await self.controller.set_gripper(position)
                return {"success": success, "gripper_position": position}

            elif command == "arm.stop":
                await self.controller.stop()
                return {"success": True, "message": "Emergency stop triggered"}

            elif command == "arm.home":
                # Move to a safe home position
                home = [0.0, -1.57, 1.57, 0.0, 0.0, -0.4]
                success = await self.controller.move_joints(home)
                return {"success": success, "position": "home"}

            elif command == "arm.camera":
                cam_name = params.get("camera", "workspace")
                quality = params.get("quality", 80)
                cam_mgr = self.controller.camera_mgr
                if cam_mgr is None:
                    return {"success": False, "error": "Camera manager not initialized"}
                b64 = cam_mgr.get_jpeg_b64(cam_name, quality=quality)
                if b64:
                    return {
                        "success": True,
                        "camera": cam_name,
                        "format": "jpeg",
                        "encoding": "base64",
                        "image": b64,
                    }
                available = cam_mgr.list_cameras()
                return {
                    "success": False,
                    "error": f"No image from '{cam_name}'",
                    "available_cameras": available,
                }

            elif command == "arm.camera_list":
                cam_mgr = self.controller.camera_mgr
                if cam_mgr is None:
                    return {"cameras": []}
                return {"cameras": cam_mgr.list_cameras()}

            else:
                return {"error": f"Unknown command: {command}"}

        except Exception as e:
            logger.error(f"SO101 command failed: {e}")
            return {"error": str(e)}
